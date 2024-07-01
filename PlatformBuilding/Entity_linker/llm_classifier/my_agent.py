import json
from typing import Dict, List
from langchain.llms import SagemakerEndpoint
from langchain.llms.sagemaker_endpoint import LLMContentHandler
from typing import Dict, List
from transformers import AutoTokenizer



class ContentHandler(LLMContentHandler):
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, prompt: str, model_kwargs: Dict) -> bytes:
        prompt = {'inputs': prompt, 'parameters': model_kwargs}
        input_str = json.dumps(prompt)
        return input_str.encode("utf-8")

    def transform_output(self, output: bytes) -> str:
        response_json = json.loads(output.read().decode("utf-8"))
        return response_json[0]["generated_text"].strip()


class ChatFormatter():
    def __init__(self, model='mistral'):
        self.model = model

    def format_message(self, message):
        if 'orca' in self.model or 'default' in self.model:
            chat_message = self.format_orca(message)
        elif 'llama3' in self.model:
            chat_message = self.format_llama3(message)
        elif 'mistral_instruct' in self.model or 'mixtral' in self.model or 'biomistral' in self.model or 'llama' in self.model:
            chat_message = self.format_mistral(message)
        elif 'zephyr' in self.model:
            chat_message = self.format_zephyr(message)
        elif 'claude' in self.model or 'llama2' in self.model or 'jurassic' in self.model:
            chat_message = self.format_claude(message)
        elif 'phi' in self.model:
            chat_message = self.format_phi(message)
        elif 'stablelm' in self.model:
            chat_message = self.format_stablelm(message)
        else:
            chat_message = self.default_format(message)

        return chat_message


    def format_zephyr(self, message):
        tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta")
        tokenizer.use_default_system_prompt = False
        message = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        return message

    def format_llama3(self, message):
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
        tokenizer.use_default_system_prompt = False
        message = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        return message

    def format_phi(self, message):
        output_message = ''
        for elem in message:
            if elem['role'] == 'user':
                output_message += 'Instruct: {} \n'.format(elem['content'])
            else:
                output_message += 'Output: {} \n'.format(elem['content'])

        output_message += 'Output:'
        return output_message

    def format_orca(self, message):
        tokenizer = AutoTokenizer.from_pretrained('microsoft/Orca-2-13b', use_fast=False)
        tokenizer.use_default_system_prompt = False
        tokenizer.chat_template = "{% if not add_generation_prompt is defined %}{% set add_generation_prompt = false %}{% endif %}{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
        message = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        return message

    def format_stablelm(self, message):
        tokenizer = AutoTokenizer.from_pretrained('stabilityai/stablelm-zephyr-3b')
        tokenizer.use_default_system_prompt = False
        message = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        return message

    def default_format(self, message):
        output_message = ''
        for elem in message:
            if elem['role'] == 'system':
                output_message += '{} \n'.format(elem['content'])
            elif elem['role'] == 'user':
                output_message += '{} \n'.format(elem['content'])
            else:
                output_message += 'Completion: {} \n'.format(elem['content'])

        output_message += 'Completion: '
        return output_message


    def format_mistral(self, message): # Doesn't support system, so we merge them with the user message}
        new_message_list = []
        for i in range(len(message)):
            new_message = {}
            if message[i]['role'] == 'system':
                new_message['role'] = 'user'
                if message[i + 1]['role'] == 'user' and i + 1 < len(message):
                    new_message['content'] = message[i]['content'] + '\n' + message[i + 1]['content']
                    new_message_list.append(new_message)

            elif message[i]['role'] == 'user':
                if message[i - 1]['role'] == 'system' and i - 1 >= 0:
                    pass
                else:
                    new_message_list.append(message[i])

            else: #message[i]['role'] == 'assistant':
                new_message_list.append(message[i])

        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
        tokenizer.use_default_system_prompt = False
        message = tokenizer.apply_chat_template(new_message_list, tokenize=False, add_generation_prompt=True)
        return message

    def format_claude(self, message):
        new_message_list = []
        for i in range(len(message)):
            new_message = {}
            if message[i]['role'] == 'system':
                new_message['role'] = 'user'
                if message[i + 1]['role'] == 'user' and i + 1 < len(message):
                    new_message['content'] = message[i]['content'] + '\n' + message[i + 1]['content']
                    new_message_list.append(new_message)

            elif message[i]['role'] == 'user':
                if message[i - 1]['role'] == 'system' and i - 1 >= 0:
                    pass
                else:
                    new_message_list.append(message[i])

            else:  # message[i]['role'] == 'assistant':
                new_message_list.append(message[i])


        output_message = ''
        for elem in new_message_list:
            if elem['role'] == 'system':
                output_message += 'Instruction: {} \n'.format(elem['content'])
            elif elem['role'] == 'user':
                output_message += 'Human: {} \n'.format(elem['content'])
            else:
                output_message += 'Assistant: {} \n'.format(elem['content'])
        output_message += 'Assistant:'

        return output_message




class MyAgent:
    def __init__(self, endpoint_name="jumpstart-dft-hf-llm-mistral-7b-instruct", model_name='mistral'):
        self.llm = SagemakerEndpoint(
            endpoint_name=endpoint_name,
            # credentials_profile_name="credentials-profile-name",
            region_name="us-east-2",
            model_kwargs={"temperature": 1e-10},
            content_handler=ContentHandler())
        self.model_name = model_name
        self.chat_templater = ChatFormatter(model=model_name)


    def print_instructions(self, prompt: str, response: str) -> None:
        bold, unbold = '\033[1m', '\033[0m'
        print(f"{bold}> Input{unbold}\n{prompt}\n\n{bold}> Output{unbold}\n{response}\n")


    def predict(self, instructions=[], model_kwargs={}, print_response=False):
        model_kwargs['return_full_text'] = False
        prompt = self.chat_templater.format_message(instructions)
        #print(prompt)
        #print(model_kwargs)
        out = self.llm.predict(prompt, **model_kwargs)
        #print('Out: {} |||||'.format(out))

        if print_response:
            self.print_instructions(prompt, out)

        return out

if __name__ == '__main__':
    chat = [
        {"role": "system", "content": "You are a helpful chatbot"},

        {"role": "user", "content": "Hello, how are you?"},

        {"role": "assistant", "content": "I'm doing great. How can I help you today?"},

        {"role": "user", "content": "I'd like to show off how chat templating works!"},

    ]

    # tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta")
    # tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.1")
    # tokenizer.use_default_system_prompt = False
    # message = tokenizer.apply_chat_template(chat, tokenize=False)
    chat = [{'role': 'system',
             'content': "You are a linguistic expert and a radiology specialist who understands everything related to CT radiology reports. \nYou can tell exactly WHAT medical entities are mentioned in an excerpt from a CT report, WHERE they were found, and infer WHAT is their condition. \n\n\n- Given an excerpt of a CT report, you must identify What medical entities are mentioned by selecting the correct alternative. \n- Please choose the most concise alternative, which is not too specific. You can select multiple alternatives separated by comma, if necessary\n- Choose the alternative(s) that doesn't point out a location, unless the medical entity referenced is a location itself\n- If you think that the correct answer is not in the alternatives, select alternative j).\n- Please just answer with the letter of the correct alternative(s) and nothing else.\n"},
            {'role': 'user', 'content': 'Have you understood the instruction?'}, {'role': 'assistant',
                                                                                  'content': 'Yes, I am an expert radiologist and will be able to answer all the questions froms the excerpts'},
            {'role': 'system',
             'content': "Excerpt from the report section 'impression': 'differential includes the sequela of prior infarct, utilizing round atelectasis with bronchiectasis, cavitary lung mass or other' "},
            {'role': 'user',
             'content': 'Based only in the excerpt and the following alternatives, What is the referenced medical entity? \n\nAlternatives:\na) cavitary lung mass \nb) bronchiectasis \nc) infarct \nd) prior infarct \ne) mass \nf) atelectasis \ng) round atelectasis \nh) lung mass \ni) cavitary \nj) None of the alternatives \n\n'},
            {'role': 'assistant', 'content': 'c), f), b), e)'}, {'role': 'system',
                                                                 'content': "Excerpt from the report section 'impression': 'irregular intermediate density structure outpouching along the lateral aspect of the aortic arch, could reflect a postoperative hematoma, although given the morphology cta of the chest is recommended to exclude underlying pseudoaneurysm' "},
            {'role': 'user',
             'content': 'Based only in the excerpt and the following alternatives, What is the referenced medical entity? \n\nAlternatives:\na) density \nb) a postoperative hematoma \nc) outpouching \nd) irregular intermediate density structure outpouching \ne) hematoma \nf) underlying pseudoaneurysm \ng) pseudoaneurysm \nh) intermediate density \ni) None of the alternatives \n\n'},
            {'role': 'assistant', 'content': 'e), g)'}, {'role': 'system',
                                                         'content': "Excerpt from the report section 'impression': 'differential includes the sequela of prior infarct, utilizing round atelectasis with bronchiectasis, cavitary lung mass or other' "},
            {'role': 'user',
             'content': 'Based only in the excerpt and the following alternatives, What is the referenced medical entity? \n\nAlternatives:\na) cavitary lung mass \nb) bronchiectasis \nc) infarct \nd) prior infarct \ne) mass \nf) atelectasis \ng) round atelectasis \nh) lung mass \ni) cavitary \nj) None of the alternatives \n\n'}]

    #chat = [{'role': 'user', 'content': 'hi'}, {'role': 'assistant', 'content': 'how are you?'}, {'role': 'user', 'content': 'Good and you?'}]
    chat_formatter = ChatFormatter(model='orca')
    mistral_message = chat_formatter.format_message(message=chat)

    chat_formatter = ChatFormatter(model='zephyr')
    zephyir_message = chat_formatter.format_message(message=chat)

    chat_formatter = ChatFormatter(model='normal')
    normal_message = chat_formatter.format_message(message=chat)

    chat_formatter = ChatFormatter(model='stablelm')
    stable_message = chat_formatter.format_message(message=chat)



    print('Mistral')
    print(mistral_message)
    print('')
    print('Zephyr')
    print(zephyir_message)
    print('')
    print('Normal')
    print(normal_message)
    print('Stable')
    print(stable_message)


    myllm = MyAgent(endpoint_name="hf-llm-mixtral-8x7b-instruct-2024-01-09-18-36-52-069", model_name='mixtral')
    instructions = [
        {"role": "user",
         "content": """
    You are a linguistic expert and a radiology specialist who understands everything related to CT radiology reports. 
    You can tell exactly WHAT medical entities are mentioned in an excerpt from a CT report, WHERE they were found, and WHAT IS THEIR CONDITION. 
    Here is an excerpt from a CT report from the lungs section:

    'central airways are patent'

    Based on the excerpt, what is the referenced medical entity? 
    a) central airways 
    b) patent 
    c) airways 
    d) None of the alternatives 

    Here are additional instructions that you must follow:
    - Please choose the most concise alternative, which is not too specific. 
    - Choose the alternative that doesn't point out a location, unless the medical entity referenced is a location itself
    - If you think that the correct answer is not in the alternatives, select alternative d).
    - Please just answer with the letter of the correct alternative and nothing else.
    """}]

    model_kwargs = {'temperature': 1e-2, 'max_new_tokens': 256, 'do_sample': False}
    out = myllm.predict(instructions=instructions, model_kwargs=model_kwargs, print_response=True)
    print(out)

#     myllm = MyAgent(endpoint_name="jumpstart-dft-hf-llm-mistral-7b",
#                     format_instructions=[])
#     instructions = """You are a linguistic expert and a radiology specialist who understands everything related to CT radiology reports.
# Here is the excerpt we have been analyzing:
#
# 'the central airways are clear'
#
# Based only in the excerpt, are you sure the following makes sense?
# Instruction: Just answer Yes or No, nothing aelse
# - One of the Medical entities referenced is: 'central airways'?
# Answer:"""
#
#     model_kwargs = {'temperature': 1e-1, 'max_new_tokens': 20, 'do_sample': False}
#     response = myllm.predict(instructions=instructions, model_kwargs=model_kwargs)
#
#     print(repr(response))
