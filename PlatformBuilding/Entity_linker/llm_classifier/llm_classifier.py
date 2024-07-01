import time
import torch
import copy
from scipy.stats import entropy
import random
import re
import numpy as np
from collections import Counter
from llm_classifier.templates import template_system, template_user, template_general_instruction


class LLMClassifier:
    def __init__(self, llm, model_name=''):
        self.llm = llm
        self.model_name = model_name
        self.dicto_alternatives = {'a': 'positive', 'b': 'negative', 'c': 'relate', 'd': 'na'}
        self.dicto_transform = {'positive': 'positive', 'negative': 'negative', 'related': 'relate', 'relate': 'relate',
                                'na': 'na', 'nan': 'na'}
        self.dicto_alternatives_backward = {v: k for k, v in self.dicto_alternatives.items()}
        # self.dicto_alternatives_backward['nan'] = 'd'
        self.tries_limit = [5, 10]


    def format_answer(self, answer):
        answer = answer.strip()
        answer = answer.split('\n')[0].strip()
        return answer

    def format_yes_no(self, answer):
        answer = 'yes' if 'yes' in answer.lower() else answer
        return answer

    def format_arbitrary_text(self, answer):
        return answer.strip()

    def query_llm(self, prompt, model_kwargs={}, print_response=False):
        answer = self.llm.predict(instructions=prompt, model_kwargs=model_kwargs, print_response=print_response)
        answer = answer.replace('<|assistant|>', '').strip()
        return answer

    def dialog_llm_user(self, instruction_prompt, format_func, model_kwargs={}):
        answer = self.query_llm(instruction_prompt, model_kwargs=model_kwargs)
        #print('Real answer: {}'.format(answer))
        answer = format_func(answer)
        #print('Formatted answer: {}'.format(answer))
        instruction_prompt.append({'role': 'assistant', 'content': answer + '\n'})

        return answer, instruction_prompt



    def rephrase(self, sentence, microbe, disease, model_kwargs={}):
        instruction_prompt = [{'role': 'system', 'content': template_general_instruction + '\n' }]
        instruction_prompt.append({'role': 'user', 'content': copy.deepcopy(template_rephrase).format(disease=disease, microbe=microbe,
                                                                                                      sentence=sentence)})
        sentence_rephrasing, instruction_prompt = self.dialog_llm_user(instruction_prompt=instruction_prompt,
                                                            format_func=self.format_arbitrary_text,
                                                            model_kwargs=model_kwargs)

        return sentence_rephrasing


    def get_relation(self, zero_shot=False, repeat_instruction=True, sentence='microbe is related to disease',
                     disease='disease', microbe='microbe',
                     examples=[], confirmation_prompt=[], last_answer_prompt=[], model_kwargs={}):
        instruction = []


        # Examples if necessary
        if not zero_shot:
            instruction += examples
        else:
            instruction.append({'role': 'system', 'content': template_system + '\n'})

        # Question
        instruction.append({'role': 'user', 'content': copy.deepcopy(template_user)
                               .format(microbe=microbe, disease=disease, evidence=sentence)})

        # Answer
        selected_relation, instruction = self.dialog_llm_user(instruction_prompt=instruction,
                                                                 format_func=self.format_answer,
                                                                 model_kwargs=model_kwargs)

        if 'negative' in selected_relation:
            selected_relation = 'negative'
        elif 'positive' in selected_relation:
            selected_relation = 'positive'
        else: # 'relate' in selected_relation:
            selected_relation = 'na'

        return selected_relation, instruction


    def voting_relations(self, sentences, disease, microbe, examples, model_kwargs=[], zero_shot=False,
                         confirmation_prompt=[], last_answer_prompt=[], repeat_instruction=False):
        selections = {'positive': 0, 'negative': 0, 'na': 0}
        example_n = 0
        total_reps = 0
        dialogues = []
        for sentence in sentences:
            model_kwargs_n = 0
            for kwargs in model_kwargs:
                verifier = False
                tries = 0
                while not verifier:
                    try:
                        selected_relation, instruction_prompt = (
                            self.get_relation(zero_shot=zero_shot, repeat_instruction=repeat_instruction,
                                              sentence=sentence, disease=disease, microbe=microbe,
                                              examples=examples, confirmation_prompt=copy.deepcopy(confirmation_prompt),
                                              last_answer_prompt=copy.deepcopy(last_answer_prompt),
                                              model_kwargs=kwargs))

                        verifier = True
                        dialogues.append(instruction_prompt)
                    except KeyError as e:
                        tries += 1
                        print('Relations error: {} | Tries: {}'.format(e, tries))
                        if tries >= self.tries_limit[0] and tries < self.tries_limit[1]:
                            kwargs['temperature'] += 1e-2
                        elif tries >= self.tries_limit[1]:  # Just random answer
                            print('Solving with random answer')
                            selected_relation = random.choice(['positive', 'negative', 'na'])
                            verifier = True

                selections[selected_relation] += 1
                model_kwargs_n += 1
                total_reps += 1


        min_th = int(total_reps / 2) + 1  # 50% + 1
        max_value = max(list(selections.values()))
        for k, v in selections.items():
            if v == max_value:
                selected_relation = k
                break

        # Entropy calculation
        values = np.array([elem for elem in selections.values()])
        H = entropy(values)

        return selected_relation, selections, H, dialogues


    def edit_prompt(self, prompt, added_text='', position='before', index=0):
        copied_prompt = copy.deepcopy(prompt)
        if position == 'before':
            copied_prompt[index]['content'] = added_text + '\n' + prompt[index]['content']
        else:
            copied_prompt[index]['content'] = prompt[index]['content'] + '\n' + added_text

        return copied_prompt



    def run(self, sentence, microbe, disease, examples=[], examples_COT=[], model_kwargs={'temperatures': 0.7}, zero_shot=True, max_tries=5,
            rephrase=False, n_rephrasings=1):


        original_sentence = copy.deepcopy(sentence)

        print('Microbe: {} | Disease: {}'.format(microbe, disease))
        print('Original Sentence: {}'.format(original_sentence))
        print('')

        self_check_scores = {'ngram': [], 'bertscore': [], 'nli': [], 'entropy_alternatives': [], 'voting_alternatives': [],
                             'entropy_COT': [], 'voting_COT': [], 'entropy_conf': [], 'voting_conf': []}

        all_selections = []
        dialogues = {}

        ####################################### Rephrasing ################################################
        if n_rephrasings > 1:
            sentences = []
            summary_index = 0

            for _ in range(n_rephrasings):
                sentence = self.rephrase(original_sentence, microbe, disease, model_kwargs={'temperature': 0.7,
                                                                                            'max_new_tokens': 70})
                sentences.append(sentence)
                print('Summary {}: {}'.format(summary_index + 1, sentence))
                summary_index += 1
            print('')
            if n_rephrasings > 1:

                sentences = [original_sentence] + sentences
            else:  # only one rephrasing, is treated as the real sentence
                original_sentence = sentences[0]

        else:
            sentences = [original_sentence]

        ########################################### Voting Alternatives ################################################
        selected_relation_alt, all_chosen_relations_alt, entropy_relations_alt, dialogues_alt = (
            self.voting_relations(sentences, disease, microbe, examples, model_kwargs=model_kwargs, zero_shot=zero_shot,
                                  confirmation_prompt=[], last_answer_prompt=[], repeat_instruction=False))

        print('Selected alternative: {}'.format(selected_relation_alt))
        print('Voting alternatives results: {}'.format(all_chosen_relations_alt))
        print('')

        # Getting entropy scores
        self_check_scores['entropy_alternatives'] = entropy_relations_alt
        self_check_scores['voting_alternatives'] = all_chosen_relations_alt

        all_selections.append(selected_relation_alt)
        dialogues['alternatives'] = dialogues_alt


        ###################################### Final selection #########################################################
        all_selections = Counter(all_selections)
        selected_relation = all_selections.most_common()
        max_val = max([elem[1] for elem in selected_relation])
        selected_relation_aux = []
        for elem in selected_relation:
            if elem[1] == max_val:
                selected_relation_aux.append(elem[0])

        if len(selected_relation_aux) > 1:
            if 'na' in selected_relation_aux:
                selected_relation = 'na'
            else:
                selected_relation = selected_relation_aux[0]
        else:
            selected_relation = selected_relation_aux[0]

        # Choosing if entropy is 0
        alt_entropy = entropy(np.array(list(all_chosen_relations_alt.values())))
        if alt_entropy == 0: # Completely sure of the answer
            return selected_relation
        else:
            return 'na'


if __name__ == '__main__':
    from my_agent import MyAgent

    sentence = 'E.coli is positively correlated with diabetes but not with pneumonia'
    microbe = 'E.coli'
    disease = 'diabetes'

    model_name = 'biomistral'
    llm = MyAgent(endpoint_name='huggingface-pytorch-tgi-inference-2024-05-02-16-40-28-306', model_name=model_name)

    classifier = LLMClassifier(llm=llm, model_name=model_name)

    out = classifier.run(sentence, microbe, disease, model_kwargs=[{'temperature': 0.7, 'max_new_tokens': 20, 'do_sample': True} for _ in range(7)], zero_shot=True,
            rephrase=False, n_rephrasings=1)

    print(out)

