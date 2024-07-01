import numpy as np
import pandas as pd
from new_release_paper.normal_db.data_loader import DataLoader
import copy
from new_release_paper.llm.my_agent import MyAgent
import random
import os

# Templates
template_general_instruction = """You are an expert microbiologist researcher that writes in a scientifically manner"""
template_specific_instruction = """Given the following four possible types of relations between a microbe and a disease:
POSITIVE: This type is used to annotate microbe-disease entity pairs with positive correlation, such as microbe will cause or aggravate the disease, the microbe will increase when disease occurs.
NEGATIVE: This type is used to annotate microbe-disease entity pairs that have a negative correlation, such as microbe can be a treatment for a disease, or microbe will decrease when disease occurs. 
RELATE: This type is used when a microbe-disease entity pair appears in the instance and they are related to each other without additional information.
NA: This type is used when a microbe-disease entity pair appears in the instance, but the relationship of these two entities has not been described as positive, negative, or relate.

Here are some examples on how to write sentences of {relation} relations.
"""

template_instruction_example = """Please write a sentence that implies a {relation} relation between microbe '{microbe}' and disease '{disease}'.
 Do not explicitly mention the type of the relation and DON'T forget to explicitly mention the microbe '{microbe}' and disease '{disease}'"""

class DatataAugmentatorRAG:
    def __init__(self, llm):
        self.llm = llm
        self.model_name = llm.model_name
        self.train_datas = self.get_train_data()
        self.silver_data = pd.read_excel('../../databases/llms-microbe-disease/data/data_MDI_SilverDataforTrain.xlsx')
        self.all_micobres = list(set(self.silver_data['MICROBE'].values.tolist()))
        self.all_diseases = list(set(self.silver_data['DISEASE'].values.tolist()))

        self.folder = 'augmentations/{}/'.format(self.model_name)
        if not os.path.exists(self.folder):
            os.makedirs(self.folder)

    def get_train_data(self):
        # New augmentations
        llm_augmentations_path = '../normal_db/augmentations/llm_augmentations.csv'
        shuffling_augmentations_path = '../normal_db/augmentations/shuffling_aumentation.csv'
        combine_augmentations_path = '../normal_db/augmentations/shuffling_llm_aumentation.csv'
        data_path = '../../databases/llms-microbe-disease/data/gold_data_corrected.csv'
        data_splits = {}
        for i in range(5):
            data_loader = DataLoader(data_path=data_path, use_gold=True, split_gold=True, k=i,
                                     llm_augmentations_path=llm_augmentations_path,
                                     shuffling_augmentations_path=shuffling_augmentations_path,
                                     combine_augmentations_path=combine_augmentations_path)

            data_splits['k{}'.format(i)] = data_loader.data

        train_data_k0 = data_splits['k0']
        train_data_k1 = data_splits['k1']
        train_data_k2 = data_splits['k2']
        train_data_k3 = data_splits['k3']
        train_data_k4 = data_splits['k4']

        train_datas = {'k0': train_data_k0, 'k1': train_data_k1, 'k2': train_data_k2, 'k3': train_data_k3,
                       'k4': train_data_k4}
        return train_datas

    def divide_by_classes(self, k=0):
        train_data = self.train_datas['k{}'.format(k)]
        positives = train_data[train_data['RELATION'] == 'positive']
        negatives = train_data[train_data['RELATION'] == 'negative']
        relates = train_data[train_data['RELATION'] == 'relate']
        nas = train_data[train_data['RELATION'] == 'na']
        return positives, negatives, relates, nas

    def format_arbitrary_text(self, answer):
        return answer.replace('\n', '').strip()

    def query_llm(self, prompt, model_kwargs={}, print_response=False):
        answer = self.llm.predict(instructions=prompt, model_kwargs=model_kwargs, print_response=print_response)
        answer = answer.replace('<|assistant|>', '').strip()
        return answer

    def dialog_llm_user(self, instruction_prompt, format_func, model_kwargs={}):
        answer = self.query_llm(instruction_prompt, model_kwargs=model_kwargs)
        answer = format_func(answer)
        instruction_prompt.append({'role': 'assistant', 'content': answer + '\n'})

        return answer, instruction_prompt


    def augment_class(self, class_df, target_relation='positive', model_kwargs={}, n_examples=20, n_augmentations=100):
        augmentations = []
        aug_index = 0

        n_examples = min(class_df.shape[0] - 5, n_examples)
        print('Extracting {} examples'.format(n_examples))

        #n_augmentations = min(3*class_df.shape[0], n_augmentations) # We can generate at the most three times the sentences
        #print('Generating {} augmentations'.format(n_augmentations))

        while len(augmentations) < n_augmentations:
            # Get examples
            instructions = [{'role':'system', 'content': template_general_instruction}]
            instructions.append({'role': 'user', 'content': template_specific_instruction})
            instructions.append({'role': 'assistant', 'content': 'Okay! I will consider thos examples and will write a good sentence'})

            target_microbe = random.choice(self.all_micobres)
            target_disease = random.choice(self.all_diseases)

            examples = class_df.sample(n=n_examples)
            for i in range(n_examples):
                row = examples.iloc[i]
                microbe = row['MICROBE']
                disease = row['DISEASE']
                relation = row['RELATION']
                evidence = row['EVIDENCE']

                instructions.append({'role': 'user', 'content': copy.deepcopy(template_instruction_example).format(microbe=microbe,
                                                                                                                   disease=disease,
                                                                                                                   relation=target_relation.upper())})

                instructions.append({'role': 'assistant', 'content': evidence})

            # Final instruction
            instructions.append(
                {'role': 'user', 'content': copy.deepcopy(template_instruction_example).format(microbe=target_microbe,
                                                                                               disease=target_disease,
                                                                                               relation=target_relation.upper())})

            augmentation, instructions = self.dialog_llm_user(instruction_prompt=instructions,
                                                                           format_func=self.format_arbitrary_text,
                                                                           model_kwargs=model_kwargs)

            print('{}) Aug: {}'.format(aug_index + 1, augmentation))

            if target_disease.lower() not in augmentation.lower() or target_microbe.lower() not in augmentation.lower():
                print('Does not have microbe or disease!')
                instructions.append({'role': 'user', 'content': 'Remeber that you MUST explicitly mention microbe '
                                                                '{} and disease {} in your answer'.format(target_microbe, target_disease)})
                instructions.append({'role': 'assistant', 'content': 'Okay, I will try again'})
                instructions.append(
                    {'role': 'user',
                     'content': copy.deepcopy(template_instruction_example).format(microbe=target_microbe,
                                                                                   disease=target_disease,
                                                                                   relation=target_relation.upper())})

                augmentation, instructions = self.dialog_llm_user(instruction_prompt=instructions,
                                                                  format_func=self.format_arbitrary_text,
                                                                  model_kwargs=model_kwargs)
                print('{}) Corrected Aug: {}'.format(aug_index + 1, augmentation))

                if target_disease.lower() not in augmentation.lower() or target_microbe.lower() not in augmentation.lower():
                    print('Correction didnt work')

                    continue

            augmentation = self.reconstruct_df(relation=target_relation, microbe=target_microbe,
                                               disease=target_disease, evidence=augmentation)
            augmentations.append(augmentation)
            aug_index += 1

        augmentations = pd.concat(augmentations, axis=0)
        return augmentations

    def reconstruct_df(self, relation, microbe, disease, evidence):
        question = 'What is the relationship between {} and {}?'.format(microbe, relation)
        df = np.array([microbe, disease, evidence, relation, question]).reshape(1, -1)
        df = pd.DataFrame(df, columns=['MICROBE', 'DISEASE', 'EVIDENCE', 'RELATION', 'QUESTIONS'])
        return df

    def run(self, n_examples=15, model_kwargs={'temperature': 0.8, 'max_new_tokens':100},
            class_augmentations={'positive': 10, 'negative': 15, 'relate': 20, 'na': 30}):

        for k in range(5):
            print('Augmentations for k={}'.format(k))
            positives, negatives, relates, nas = self.divide_by_classes(k=k)

            print('Augmenting positives \n')
            positive_augmentations = self.augment_class(positives, target_relation='positive', n_examples=n_examples,
                                           model_kwargs=model_kwargs, n_augmentations=class_augmentations['positive'])

            print('Augmenting negatives \n')
            negative_augmentations = self.augment_class(negatives, target_relation='negative', n_examples=n_examples,
                                                        model_kwargs=model_kwargs, n_augmentations=class_augmentations['negative'])

            print('Augmenting relates \n')
            relate_augmentations = self.augment_class(relates, target_relation='relate', n_examples=n_examples,
                                                        model_kwargs=model_kwargs, n_augmentations=class_augmentations['relate'])

            print('Augmenting nas \n')
            na_augmentations = self.augment_class(nas, target_relation='na', n_examples=n_examples,
                                                        model_kwargs=model_kwargs, n_augmentations=class_augmentations['na'])

            all_augmentations_k = pd.concat([positive_augmentations, negative_augmentations, relate_augmentations, na_augmentations], axis=0)
            all_augmentations_k = all_augmentations_k.sample(frac=1)
            all_augmentations_k.to_csv('{}llm_RAG_k{}.csv'.format(self.folder, k))



if __name__ == '__main__':
    # LLM
    endpoint_name = 'jumpstart-dft-hf-llm-mixtral-8x7b-instruct'
    model_name = 'mixtral'
    myllm = MyAgent(endpoint_name=endpoint_name, model_name=model_name)

    data_augmentator = DatataAugmentatorRAG(llm=myllm)
    data_augmentator.run(class_augmentations={'positive': 30, 'negative': 50, 'relate': 80, 'na': 100},
                         model_kwargs={'temperature': 0.8, 'max_new_tokens': 100})