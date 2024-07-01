from dataclasses import dataclass, field
import comet_ml
import os

# upgrade flash attention here
try:
    os.system("pip install flash-attn --no-build-isolation --upgrade")
except:
    print("flash-attn failed to install")


from typing import Optional
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
    default_data_collator,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    HfArgumentParser,
)
from transformers import EarlyStoppingCallback
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, setup_chat_format
from datasets import load_from_disk
import torch

import bitsandbytes as bnb
from huggingface_hub import login
from peft import (
        get_peft_model,
        LoraConfig,
        TaskType,
        prepare_model_for_kbit_training,
    )




# COPIED FROM https://github.com/artidoro/qlora/blob/main/qlora.py
def find_all_linear_names(model):
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, bnb.nn.Linear4bit):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


def create_peft_model(model, gradient_checkpointing=True, bf16=True):
    from peft import (
        get_peft_model,
        LoraConfig,
        TaskType,
        prepare_model_for_kbit_training,
    )
    from peft.tuners.lora import LoraLayer

    # get lora target modules
    modules = find_all_linear_names(model)
    print(f"Found {len(modules)} modules to quantize: {modules}")

    peft_config = LoraConfig(
        r=64,
        lora_alpha=16,
        target_modules=modules,
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, peft_config)

    model.print_trainable_parameters()
    return model


@dataclass
class ScriptArguments:
    model_id: str = field(
        metadata={
            "help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."
        },
    )
    hub_name: str = field(default='my_mistral')
    response_template: str = field(default='[/INST]')

    instruction_base_model: bool = field(default=False)
    full_instruct_model: bool = field(default=False)
    save_model: bool = field(default=False)

    dataset_path: str = field(
        metadata={"help": "Path to the preprocessed and tokenized dataset."},
        default=None,
    )

    val_dataset_path: str = field(
        metadata={"help": "Path to the preprocessed validation data and tokenized dataset."},
        default=None,
    )
    hf_token: Optional[str] = field(default=None, metadata={"help": "Hugging Face token for authentication"})
    trust_remote_code: bool = field(
        metadata={"help": "Whether to trust remote code."},
        default=False,
    )
    use_flash_attn: bool = field(
        metadata={"help": "Whether to use Flash Attention."},
        default=False,
    )
    merge_adapters: bool = field(
        metadata={"help": "Wether to merge weights for LoRA."},
        default=False,
    )


def training_function(script_args, training_args):
    # load dataset    
    train_dataset = load_from_disk(script_args.dataset_path)
    validation_dataset = load_from_disk(script_args.val_dataset_path)

    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_id,
        use_cache=False if training_args.gradient_checkpointing else True,  # this is needed for gradient checkpointing
        device_map="auto",
    )


    peft_config = LoraConfig(
        r=64,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    # Loading tokenizer
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_id)

    
    # Processing dataset
    def format_func(example):
        out = tokenizer.apply_chat_template(example['message'], tokenize=False)
        out = {'message': out}
        return out
        
    # Setup chat format
    response_template = script_args.response_template
    if script_args.instruction_base_model:
        response_template = "[/INST]"#'assistant<|end_header_id|>'#"[/INST]"#"\n<|assistant|>\n" #'<|endoftext|>\n<|assistant|>'
        tokenizer.padding_side = "right"
        tokenizer.pad_token = tokenizer.eos_token
    else:
        model, tokenizer = setup_chat_format(model, tokenizer)
        response_template = '<|im_start|>assistant'
        tokenizer.padding_side = "right"
    
    
    train_dataset = train_dataset.map(format_func, remove_columns=train_dataset.column_names)
    validation_dataset = validation_dataset.map(format_func, remove_columns=validation_dataset.column_names)
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)


    # Create Trainer instance
    if script_args.full_instruct_model:
        trainer = SFTTrainer(
            model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
            neftune_noise_alpha=5,
            max_seq_length=1024,
            dataset_text_field='message',
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
            peft_config=peft_config)
    else:
        trainer = SFTTrainer(
            model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
            neftune_noise_alpha=5,
            max_seq_length=1024,
            dataset_text_field='message',
            data_collator=collator,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
            peft_config=peft_config)

    # Start training
    trainer.train()
    trainer.push_to_hub(script_args.hub_name)
    sagemaker_save_dir = "/opt/ml/model/"
    if script_args.merge_adapters:
        # merge adapter weights with base model and save
        # save int 4 model
        trainer.model.save_pretrained(training_args.output_dir, safe_serialization=False)
        # clear memory
        del model
        del trainer
        torch.cuda.empty_cache()

        from peft import AutoPeftModelForCausalLM

        # load PEFT model in fp16
        model = AutoPeftModelForCausalLM.from_pretrained(
            training_args.output_dir,
            low_cpu_mem_usage=True,
            #torch_dtype=torch.float16,
        )
        # Merge LoRA and base model and save
        model = model.merge_and_unload()
        model.save_pretrained(sagemaker_save_dir, safe_serialization=True, max_shard_size="2GB")
        if script_args.save_model:
            model.push_to_hub(script_args.hub_name)
    else:
        #trainer.model.save_pretrained(sagemaker_save_dir, safe_serialization=True)
        trainer.save_model(sagemaker_save_dir)
        if script_args.save_model:
            trainer.push_to_hub(script_args.hub_name)

    # save tokenizer for easy inference
    tokenizer = AutoTokenizer.from_pretrained(script_args.model_id)
    tokenizer.save_pretrained(sagemaker_save_dir)
    if script_args.save_model:
        tokenizer.push_to_hub(script_args.hub_name)


def main():

    parser = HfArgumentParser([ScriptArguments, TrainingArguments])
    script_args, training_args = parser.parse_args_into_dataclasses()

    # set seed
    set_seed(training_args.seed)

    # login to hub
    token = script_args.hf_token if script_args.hf_token else os.getenv("HF_TOKEN", None)
    if token:
        print(f"Logging into the Hugging Face Hub with token {token[:10]}...")
        login(token=token)

    # run training function
    training_function(script_args, training_args)


if __name__ == "__main__":
    main()