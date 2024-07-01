from vector_db.vector_db import MyVectorDB
from prompt_getter import PromptGetter
from llm.my_agent import MyAgent
from normal_db.data_loader import DataLoader
import copy
import os
import json
import time
from templates import template_system, template_user, template_general_instruction


class LLMEvaluator:
    def __init__(self, llm, prompt_getter, data_loader, k=None, from_index=True):
        self.llm = llm
        self.k = k
        self.prompt_getter = prompt_getter
        self.db_name = self.prompt_getter.db_name
        self.data_loader = data_loader
        self.len_data = len(self.data_loader.data)

        # Reduce to three classes
        data = self.data_loader.data
        data['RELATION'] = data['RELATION'].replace(['relate'], 'na')

        self.data_loader.data = data

        self.results_folder = 'results/{}/'.format(self.llm.model_name)
        self.results = []
        self.dialogues = {}
        self.index = 0
        self.from_index = from_index
        if not os.path.exists(self.results_folder):
            os.makedirs(self.results_folder)

    def get_examples(self, sentence, microbe, disease, modality='sentence', n_neighbors=5,
                     use_mmr=False, use_reranker=False, max_COT=None):

        if modality != 'sentence':
            # Get keywords
            instruction = []
            instruction.append({'role': 'system', 'content': copy.deepcopy(template_system) + '\n'})

            instruction.append(
                {'role': 'user', 'content': copy.deepcopy(template_user).format(evidence=sentence, microbe=microbe,
                                                                                          disease=disease) + '\n'})

            keywords, instruction = self.prompt_getter.dialog_llm_user(instruction_prompt=instruction,
                                                                       format_func=self.prompt_getter.format_arbitrary_text,
                                                                       model_kwargs={'temperature': 1e-3,
                                                                                     'max_new_tokens': 20})
            keywords = keywords.split('\n')[0].strip()
            # keywords = keywords.replace(microbe, '').replace(disease, '')
        else:
            keywords = ''

        examples = self.prompt_getter.get_examples(sentence=sentence, modality=modality,
                                                                 keywords=keywords,
                                                                 disease=disease, microbe=microbe,
                                                                 n_neighbors=n_neighbors,
                                                                 use_mmr=use_mmr, use_reranker=use_reranker,
                                                                 max_COT=max_COT)
        return examples


    def get_model_kwargs(self, temperatures=[1e-3, 1e-2, 1e-1, 0.3]):
        model_kwargs = []
        for temp in temperatures:
            model_kwargs.append({'temperature': temp, 'max_new_tokens': 20, 'do_sample':True})
        return model_kwargs


    def save_params(self, examples_modality, temperatures, zero_shot, rephrase, n_rephrasings,
                    use_COT=False, use_conf=False, use_alternatives=False):
        if n_rephrasings > 1:
            run_name = '{}_{}_examples_{}_temps{}_zs{}_re{}_{}'.format(self.llm.model_name, self.db_name,
                                                                         examples_modality, len(temperatures),
                                                                         int(zero_shot), int(rephrase), n_rephrasings)
        else:
            run_name = '{}_{}_examples_{}_temps{}_zs{}_re{}'.format(self.llm.model_name, self.db_name, examples_modality,
                                                                    len(temperatures), int(zero_shot), int(rephrase))

        if use_alternatives:
            run_name += '_ALT'
        if use_COT:
            run_name += '_COT'
        if use_conf:
            run_name += '_CONF'


        self.run_folder = '{}{}/'.format(self.results_folder, run_name)
        print('save_folder: {}'.format(self.run_folder))
        if not os.path.exists(self.run_folder):
            os.makedirs(self.run_folder)

        # Params saving
        with open('{}params.json'.format(self.run_folder, run_name), 'w') as f:
            json.dump({'examples_modality': examples_modality,
                       'temperatures': temperatures,
                       'zero_shot': zero_shot,
                       'rephrase': rephrase,
                       'db': self.db_name,
                       'n_rephrasings': n_rephrasings,
                       'k': self.k,
                       'use_COT': use_COT,
                       'use_alternatives': use_alternatives,
                       'use_conf': use_conf,
                       'model': self.llm.model_name}, f)

        # Tags
        tags = [self.llm.model_name, self.db_name]
        if zero_shot:
            tags.append('zero_shot')
        if rephrase:
            tags.append('rephrase')
        if len(temperatures) > 1:
            tags.append('temps')
        tags.append('examples_{}'.format(examples_modality))
        if n_rephrasings > 1:
            tags.append('rephrasings')
        if k != None:
            tags.append('k{}'.format(self.k))

        return run_name, tags


    def evaluate(self, examples_modality='sentence', n_neighbors=5,
                     use_mmr=False, use_reranker=False, temperatures=[1e-3, 1e-2, 1e-1, 0.3], zero_shot=False,
                 rephrase=True, n_rephrasings=1, use_COT=True, use_conf=True, use_alternatives=True):

        run_name, experiment_tags = self.save_params(examples_modality, temperatures, zero_shot, rephrase, n_rephrasings,
                                                     use_COT=use_COT, use_conf=use_conf, use_alternatives=use_alternatives)

        if self.from_index:
            try:
                with open('{}index.json'.format(self.run_folder), 'r') as f:
                    self.index = json.load(f)['index']

                with open('{}results.json'.format(self.run_folder), 'r') as f:
                    self.results = json.load(f)

                with open('{}dialogues.json'.format(self.run_folder), 'r') as f:
                    self.dialogues = json.load(f)

                print('Saved index: {}'.format(self.index))
                print('Success!!')
            except:
                self.index = 0

        for i in range(self.index, self.len_data):
            print('Analyzing {}|{}'.format(i + 1, self.len_data))

            # Getting the evaluation sentence with related information
            sentence = self.data_loader.get_evidence(index=i)
            disease = self.data_loader.get_disease(index=i)
            microbe = self.data_loader.get_microbe(index=i)
            gt_relation = self.data_loader.get_relation(index=i)

            # Get examples from the vector db
            examples = self.get_examples(modality=examples_modality, n_neighbors=n_neighbors,
                                              use_mmr=use_mmr, use_reranker=use_reranker,
                                              sentence=sentence, microbe=microbe, disease=disease)


            # Get different model characteristics varying temperatures
            model_kwargs = self.get_model_kwargs(temperatures=temperatures)
            print(model_kwargs, zero_shot, rephrase, n_rephrasings)
            print(sentence)
            print(microbe, disease)
            print('------')

            # Get the answer
            init_time = time.time()
            predicted_relation, rephrase_sentence, self_check_scores, dialogues = self.prompt_getter.run(sentence, microbe,
                                                                                              disease,
                                                                                              examples, [],
                                                                                              model_kwargs, zero_shot,
                                                                                              rephrase=rephrase,
                                                                                              n_rephrasings=n_rephrasings)
            prompt_time = time.time() - init_time


            # Saving results
            self.results.append({'sentence': sentence, 'microbe': microbe, 'disease': disease, 'GT': gt_relation,
                                 'prediction': predicted_relation, 'prompt_time': prompt_time,
                                 'self_check_scores': self_check_scores})

            self.dialogues[i] = dialogues

            self.index = i + 1


            with open('{}results.json'.format(self.run_folder), 'w') as f:
                json.dump(self.results, f)

            with open('{}index.json'.format(self.run_folder), 'w') as f:
                json.dump({'index': self.index}, f)

            with open('{}dialogues.json'.format(self.run_folder), 'w') as f:
                json.dump(self.dialogues, f)

            print('------------------ GT:{} | Predicted: {} -------------------'.format(gt_relation, predicted_relation))



if __name__ == '__main__':
    for k in range(5):
        print('*'*50 + 'k={}'.format(k) + '*'*50)
        # Model parameters
        endpoint_name = 'huggingface-pytorch-tgi-inference-2024-05-02-17-01-54-722'
        model_name = 'biomistral_FINALL'

        # DB parameters
        data_path = 'initial_db/gold_data_corrected.csv'
        use_gold = True
        split_gold = True
        db_name = 'gold_data_k{}'.format(k)

        use_COT = False
        use_conf = False
        use_alternatives = True


        #LLM
        myllm = MyAgent(endpoint_name=endpoint_name, model_name=model_name)

        # Vector db
        my_vector_db = MyVectorDB(chroma_collection=db_name, model_name=model_name, kw_file='',
                                  db_dir='vector_db/vector_db_files/')  # silver_data, silver_data_masked

        # Data Loader
        data_loader = DataLoader(data_path=data_path, use_gold=use_gold, split_gold=split_gold, k=k)

        # Prompt Getter
        prompt_getter = PromptGetter(vector_db=my_vector_db, llm=myllm, model_name=model_name, device='cpu') # For COT prompts

        # Evaluation
        llm_evaluator = LLMEvaluator(llm=myllm, prompt_getter=prompt_getter, data_loader=data_loader, from_index=True)

        llm_evaluator.evaluate(examples_modality='sentence', n_neighbors=15,
                               use_mmr=False, use_reranker=True, temperatures=[0.7]*7, zero_shot=True,
                               rephrase=False, n_rephrasings=1, use_COT=use_COT, use_conf=use_conf, use_alternatives=use_alternatives)


