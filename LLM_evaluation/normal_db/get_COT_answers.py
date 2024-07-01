import numpy as np
import torch
from templates_COT import template_general_instruction, template_specific_instruction, template_question_train, template_excerpt
from new_release_paper.llm.my_agent import MyAgent
from new_release_paper.normal_db.data_loader import DataLoader
import copy
import re
import time
import json


class GetCOT:
    def __init__(self, llm, data_loader):
        self.llm = llm
        self.model_name = self.llm.model_name
        self.data_loader = data_loader
        self.len_data = len(self.data_loader.data)
        self.index = 0
        self.all_alternatives = ['positive', 'negative', 'relate', 'na']
        try:
            self.dicto_COT_answer = torch.load('COT_prompts_{}.pkl'.format(self.model_name))
            print('Successfully loaded Dictionary')
        except:
            self.dicto_COT_answer = {}

        try:
            with open('COT_{}_index.json'.format(self.model_name), 'r') as f:
                self.index = json.load(f)['index']
            print('Successfully loaded Index')

        except:
            self.index = 0


    def query_llm(self, prompt, model_kwargs={}, print_response=False):
        answer = self.llm.predict(instructions=prompt, model_kwargs=model_kwargs, print_response=print_response)
        answer = answer.replace('<|assistant|>', '').strip()
        return answer
    
    def dialog_llm_user(self, instruction_prompt, format_func, model_kwargs={}):
        answer = self.query_llm(instruction_prompt, model_kwargs=model_kwargs)
        answer = format_func(answer)
        instruction_prompt.append({'role': 'assistant', 'content': answer + '\n'})

        return answer, instruction_prompt

    def format_arbitrary_text(self, answer):
        return answer.strip()

    def get_CoT_answers(self):
        model_kwargs = {'temperature': 1e-2, 'max_new_tokens': 200}
        for i in range(self.index, self.len_data):
            print('Analyzing {}|{}'.format(i + 1, self.len_data))
            instruction = []

            # Getting the evaluation sentence with related information
            sentence = self.data_loader.get_evidence(index=i)
            disease = self.data_loader.get_disease(index=i)
            microbe = self.data_loader.get_microbe(index=i)
            gt_relation = self.data_loader.get_relation(index=i)

            other_alternatives = copy.deepcopy(self.all_alternatives)
            other_alternatives.remove(gt_relation)

            #if gt_relation != 'na':
            #    continue

            # Templates
            correct_COT = False
            repeats = 0
            while not correct_COT:
                instruction.append({'role': 'system', 'content': copy.deepcopy(template_general_instruction) + '\n' +
                                    copy.deepcopy(template_specific_instruction)})

                instruction.append({'role': 'user', 'content': copy.deepcopy(template_excerpt).format(sentence=sentence) + '\n'
                                    + copy.deepcopy(template_question_train).format(microbe=microbe, disease=disease,
                                                                              relation=gt_relation,
                                                                              other_alternatives=', '.join(other_alternatives),
                                                                              alt1=other_alternatives[0],
                                                                              alt2=other_alternatives[1],
                                                                              alt3=other_alternatives[2])})


                answer, instruction = self.dialog_llm_user(instruction_prompt=instruction,
                                                                format_func=self.format_arbitrary_text,
                                                                model_kwargs=model_kwargs)

                print('#' * 100)
                print('ORIGINAL ANSWER: {}'.format(instruction[-1]))
                #print('#'*100)

                #for elem in instruction:
                #    print('{}: {}'.format(elem['role'], elem['content']))
                index_final_answer = answer.lower().find('therefore the correct answer')
                if index_final_answer == -1:
                    index_final_answer = answer.lower().find('therefore, the correct answer')
                    if index_final_answer == -1:
                        index_final_answer = answer.lower().find('therefore the correct alternative')
                        if index_final_answer == -1:
                            index_final_answer = answer.lower().find('therefore, the correct alternative')
                            if index_final_answer == -1:
                                index_final_answer = answer.lower().find('therefore the correct relation')
                                if index_final_answer == -1:
                                    index_final_answer = answer.lower().find('therefore, the correct relation')

                    if index_final_answer == -1:
                        print('Failed Attempt \n')
                        instruction.append({'role': 'user', 'content': 'Please FOLLOW THE PATTERN AND THE INSTRUCTION JUST AS MENTIONED ABOVE. Now, try again'})
                        instruction.append({'role': 'assistant', 'content': 'Okay, I will try again and follow the instructions just as mentioned'})
                        repeats += 1
                        if repeats == 2:
                            correct_COT = True
                    else:
                        correct_COT = True
                        pre_conclusion = answer[:index_final_answer].strip().lower()
                        print(pre_conclusion)
                        try:
                            middle = re.search('step1: (.*)\n', pre_conclusion.lower()).group(1)
                            pre_conclusion = pre_conclusion.replace(middle, 'explanation')
                        except AttributeError:
                            pass

                        print('-----------')
                        print(pre_conclusion)
                        print('-----------')


                        post_conclusion = answer[index_final_answer:].strip()
                        possibilities = ["positive", "negative", "relate", " na ", "'positive'", "'negative'",
                                         "'relate'", '"positive"', '"negative"', '"relate"', " 'na' ", 'na.', 'na\n', "'na'", '"na"', ' na']
                        dicto_originals = {"positive": 'positive', "negative": 'negative', "relate": 'relate',
                                           " na ": 'na', ' na': 'na', '"positive"':'positive', '"negative"': 'negative', '"relate"':'relate',
                                           "'positive'": 'positive', "'negative'": 'negative', "'relate'": 'relate',
                                           " 'na' ": 'na', "'na'":'na', '"na"':'na',
                                           'na.': 'na', 'na\n': 'na'}

                        pattern = r'\b(?:' + '|'.join(map(re.escape, possibilities)) + r')\b'
                        selected_relation = re.findall(pattern, post_conclusion.lower().replace("'", ''))
                        print('POST CONCLUSION: {}'.format(post_conclusion.lower().replace("'", '')))
                        print(selected_relation)
                        if len(selected_relation) > 0:
                            try:
                                selected_relation = dicto_originals[selected_relation[0]]
                            except KeyError:
                                print('Key Error, the LLM answer was: {}'.format(selected_relation[0]))

                            if selected_relation == gt_relation:
                                print('Coincidence')
                                reformatted_answer = pre_conclusion + '\n' "Therefore, the correct alternative is {}".format(gt_relation)
                                print('Reformatted answer: {}'.format(reformatted_answer))
                                self.dicto_COT_answer[(sentence, disease, microbe, gt_relation)] = reformatted_answer
                                torch.save(self.dicto_COT_answer, 'COT_prompts_{}.pkl'.format(self.model_name))
                            else:
                                print('The LLM chose: {}, even when the correct answer was: {}'.format(selected_relation, gt_relation))
                                instruction.append({'role': 'user',
                                                    'content': 'Please Remember that the correct alternative is {}, change your reasoning considering this. Now, try again'.format(gt_relation)})
                                instruction.append({'role': 'assistant',
                                                    'content': 'Okay, I will try again and create a resoning process to conclude that the correct alternative is {}'.format(gt_relation)})
                                correct_COT = False
                        else:
                            print('Answer was not found!')
                            instruction.append({'role': 'user',
                                                'content': 'Please Remember that you must end your reasoning process with: Therefore, the correct alternative is .... . Now, try again'})
                            instruction.append({'role': 'assistant',
                                                'content': 'Okay, I will try again and create a resoning process to conclude with that phrase'})
                            correct_COT = False

            self.index = i
            with open('COT_{}_index.json'.format(self.model_name), 'w') as f:
                json.dump({'index': self.index}, f)






if __name__ == '__main__':
    endpoint_name = 'huggingface-pytorch-tgi-inference-2024-01-31-09-20-43-564'#'jumpstart-dft-hf-llm-mixtral-8x7b-instruct' #"hf-llm-mixtral-8x7b-instruct-2024-01-21-13-24-02-325"
    model_name = 'orca13b'

    data_path = '../../databases/llms-microbe-disease/data/gold_data_corrected.csv'
    use_gold = True
    split_gold = False

    llm_augmentations_path = 'augmentations/llm_augmentations.csv'
    shuffling_augmentations_path = 'augmentations/shuffling_aumentation.csv'
    combine_augmentations_path = 'augmentations/shuffling_llm_aumentation.csv'
    data_loader = DataLoader(data_path=data_path, llm_augmentations_path=llm_augmentations_path, shuffling_augmentations_path=shuffling_augmentations_path,
                             combine_augmentations_path=combine_augmentations_path,
                             use_gold=use_gold, split_gold=split_gold, k=0)

    myllm = MyAgent(endpoint_name=endpoint_name, model_name=model_name)

    tester = GetCOT(llm=myllm, data_loader=data_loader)

    tester.get_CoT_answers()