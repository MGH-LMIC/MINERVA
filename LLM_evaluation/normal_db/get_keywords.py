import numpy as np
import sys
import torch
from templates_keywords import template_general_instruction, template_specific_instruction, template_excerpt, template_get_keywords
from new_release_paper.llm.my_agent import MyAgent
from new_release_paper.normal_db.data_loader import DataLoader
import copy
import time



class KeywordsGetter:
    def __init__(self, llm, data_loader):
        self.llm = llm
        self.model_name = self.llm.model_name
        self.data_loader = data_loader
        self.len_data = len(self.data_loader.data)
        self.all_alternatives = ['positive', 'negative', 'relate', 'na']
        try:
            self.dicto_keywords_answer = torch.load('Keywords_prompts_{}.pkl'.format(self.model_name))
            self.index = len(self.dicto_keywords_answer) - 1
        except:
            self.index = 0
            self.dicto_keywords_answer = {}

    def query_llm(self, prompt, model_kwargs={}, print_response=False):
        answer = self.llm.predict(instructions=prompt, model_kwargs=model_kwargs)
        return answer

    def dialog_llm_user(self, instruction_prompt, format_func, model_kwargs={}):
        answer = self.query_llm(instruction_prompt, model_kwargs=model_kwargs)
        answer = format_func(answer)
        instruction_prompt.append({'role': 'assistant', 'content': answer + '\n'})

        return answer, instruction_prompt

    def format_arbitrary_text(self, answer):
        return answer.strip()

    def get_keywords(self):
        model_kwargs = {'temperature': 1e-3, 'max_new_tokens': 20}

        for i in range(self.index, self.len_data):
            print('Analyzing {}|{}'.format(i + 1, self.len_data))
            instruction = []

            # Getting the evaluation sentence with related information
            sentence = self.data_loader.get_evidence(index=i)
            disease = self.data_loader.get_disease(index=i)
            microbe = self.data_loader.get_microbe(index=i)
            gt_relation = self.data_loader.get_relation(index=i)

            # Templates
            instruction.append({'role': 'system', 'content': copy.deepcopy(template_general_instruction) + '\n' +
                                                             copy.deepcopy(template_specific_instruction)})

            instruction.append(
                {'role': 'user', 'content': copy.deepcopy(template_excerpt).format(sentence=sentence) + '\n'
                                            + copy.deepcopy(template_get_keywords).format(microbe=microbe, disease=disease)})

            answer, instruction = self.dialog_llm_user(instruction_prompt=instruction,
                                                       format_func=self.format_arbitrary_text,
                                                       model_kwargs=model_kwargs)

            print('Original Answer: {}'.format(answer))
            keywords = answer.split('\n')[0].strip()
            keywords = keywords.replace(disease, '').replace(microbe, '')


            if keywords != None:
                self.dicto_keywords_answer[(sentence, disease, microbe, gt_relation)] = keywords
            else:
                self.dicto_keywords_answer[(sentence, disease, microbe, gt_relation)] = 'Error'

            print('Sentence: {}'.format(sentence))
            print('Disease: {} | Microbe: {}'.format(disease, microbe))
            print('Relation: {}'.format(gt_relation))
            print('Selected Keywords: {}'.format(keywords))
            print('--------------------------------------------')
            print('')

            torch.save(self.dicto_keywords_answer, 'Keywords_prompts_{}.pkl'.format(self.model_name))

    def edit_keywords(self):
        for key, values in self.dicto_keywords_answer.items():
            print(key)

if __name__ == '__main__':
    endpoint_name = 'huggingface-pytorch-tgi-inference-2024-01-31-09-20-43-564'#"hf-llm-mixtral-8x7b-instruct-2024-01-09-18-36-52-069"
    model_name = 'orca13b'

    data_path = '../../databases/llms-microbe-disease/data/gold_data_corrected.csv'
    use_gold = True
    split_gold = False

    llm_augmentations_path = 'augmentations/llm_augmentations.csv'
    shuffling_augmentations_path = 'augmentations/shuffling_aumentation.csv'
    combine_augmentations_path = 'augmentations/shuffling_llm_aumentation.csv'
    data_loader = DataLoader(data_path=data_path, llm_augmentations_path=llm_augmentations_path,
                             shuffling_augmentations_path=shuffling_augmentations_path,
                             combine_augmentations_path=combine_augmentations_path,
                             use_gold=use_gold, split_gold=split_gold, k=0)

    myllm = MyAgent(endpoint_name=endpoint_name, model_name=model_name)

    tester = KeywordsGetter(llm=myllm, data_loader=data_loader)

    tester.get_keywords()
    #tester.edit_keywords()