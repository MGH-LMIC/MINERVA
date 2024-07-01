import pandas as pd
from new_release_paper.llm.my_agent import MyAgent
import numpy as np
import copy
from new_release_paper.normal_db.data_loader import DataLoader


class DataAugmentator:
    template_general_instruction = """You are an expert microbiologist who given an excerpt from a research paper can easily 
    identify the type of relation between a microbe and a disease. Doesn't create new information, and is completely faithful to the information provided."""

    def __init__(self, data_path, myllm):
        self.data = pd.read_csv(data_path)
        self.llm = myllm
        self.relations = self.data['RELATION']
        self.relations_counts = self.relations.value_counts().to_dict()
        print('Relations counts')
        print(self.relations_counts)

    def format_arbitrary_text(self, answer):
        return answer.strip()

    def query_llm(self, prompt, model_kwargs={}, print_response=False):
        answer = self.llm.predict(instructions=prompt, model_kwargs=model_kwargs, print_response=print_response)
        answer = answer.replace('<|assistant|>', '').strip()
        return answer

    def dialog_llm_user(self, instruction_prompt, format_func, model_kwargs={}):
        answer = self.query_llm(instruction_prompt, model_kwargs=model_kwargs)
        answer = format_func(answer)
        instruction_prompt.append({'role': 'assistant', 'content': answer + '\n'})

        return answer, instruction_prompt

    def augment_df(self, df, diseases, microbes, n_augmentations=0):
        new_df = []
        for i in range(n_augmentations):
            row_sample = df.sample(n=1, axis=0)
            old_evidence = row_sample['EVIDENCE'].values[0]
            old_disease = row_sample['DISEASE'].values[0]
            old_microbe = row_sample['MICROBE'].values[0]
            old_question = row_sample['QUESTIONS'].values[0]

            disease_sample = np.random.choice(diseases, size=1)[0]
            microbe_sample = np.random.choice(microbes, size=1)[0]

            # Updates
            evidence = old_evidence.replace(old_disease, disease_sample).replace(old_microbe, microbe_sample)
            question = old_question.replace(old_disease, disease_sample).replace(old_microbe, microbe_sample)

            new_row = copy.deepcopy(row_sample)
            new_row['EVIDENCE'] = evidence
            new_row['QUESTIONS'] = question
            new_row['DISEASE'] = disease_sample
            new_row['MICROBE'] = microbe_sample
            new_df.append(new_row)

        new_df = pd.concat(new_df, axis=0)

        return new_df


    def augment_with_shuffling(self, data, relations_counts):
        microbes = np.array(list(set(list(data['MICROBE'].values))))
        diseases = np.array(list(set(list(data['DISEASE'].values))))

        na_values = data[data['RELATION'] == 'na']
        relate_values = data[data['RELATION'] == 'relate']
        negative_values = data[data['RELATION'] == 'negative']
        positive_values = data[data['RELATION'] == 'positive']

        # We will try to igualate positives
        samples_negatives = relations_counts['positive'] - relations_counts['negative']
        samples_relate = relations_counts['positive'] - relations_counts['relate']
        samples_na = relations_counts['positive'] - relations_counts['na']

        # Creating augmentations
        new_negative = self.augment_df(negative_values, diseases, microbes, samples_negatives)
        new_relate = self.augment_df(relate_values, diseases, microbes, samples_relate)
        new_na = self.augment_df(na_values, diseases, microbes, samples_na)

        return new_negative, new_relate, new_na

    def rephrase(self, sentence, microbe, disease, n_answers, template_rephrase, model_kwargs={}, instruction_prompt=[]):
        instruction_prompt.append({'role': 'user', 'content': copy.deepcopy(template_rephrase).format(disease=disease, microbe=microbe,
                                                                                                      sentence=sentence,
                                                                                                      n_answers=n_answers)})
        sentence_rephrasing, instruction_prompt = self.dialog_llm_user(instruction_prompt=instruction_prompt,
                                                            format_func=self.format_arbitrary_text,
                                                            model_kwargs=model_kwargs)

        return sentence_rephrasing, instruction_prompt

    def augment_with_llm(self, data, microbes, diseases, augmentations_per_class={'positive': 2, 'negative': 3, 'na':5, 'relate':4},
                         template='', model_kwargs={}):
        new_df = []
        for i in range(len(data)):
            row = data.iloc[i, :].to_frame().transpose()
            relation = row['RELATION'].values[0]
            microbe = row['MICROBE'].values[0]
            disease = row['DISEASE'].values[0]
            sentence = row['EVIDENCE'].values[0]
            n_augmentations = augmentations_per_class[relation]
            repeat = True
            total_reps = 0
            temperature = model_kwargs['temperature']
            max_new_tokens = model_kwargs['max_new_tokens']
            instruction_prompt = [{'role': 'system', 'content': self.template_general_instruction + '\n' }]
            print('Sentence {}/{} : \n Original: {}'.format(i+1, len(data), sentence))
            while repeat:
                temperature_aux = temperature
                rephrasing, instruction_prompt = self.rephrase(sentence=sentence, microbe=microbe, disease=disease,
                                                               instruction_prompt=instruction_prompt,
                                               template_rephrase=template, model_kwargs={'temperature': temperature_aux,
                                                                                         'max_new_tokens': max_new_tokens},
                                           n_answers=n_augmentations)
                print('Rephrasings ({} | n:{}): \n {}'.format(relation, n_augmentations, rephrasing))

                rephrasing = rephrasing.split('\n')
                rephrasing = [elem for elem in rephrasing if len(elem.strip()) > 3]
                print('Len: {}'.format(len(rephrasing)))
                len_answers = len(rephrasing)
                microbe_in_sentences = sum([1 for elem in rephrasing if microbe.lower() in elem.lower()])
                disease_in_sentences = sum([1 for elem in rephrasing if disease.lower() in elem.lower()])

                #if len_answers != n_augmentations:
                #    repeat = False
                #    print('Not all sentences!')
                #    instruction_prompt.append({'role': 'system', 'content': 'Please remember that you have to give {} different examples, try again. Just give the answers and nothing else:'.format(n_augmentations)})
                if microbe_in_sentences != n_augmentations:
                    repeat = True
                    print('Not all Microbes!')
                    instruction_prompt.append({'role': 'system', 'content': 'Please remember to explicitly mention the microbe {} in all the sentences, try again. Just give the answers and nothing else:'.format(microbe)})
                elif disease_in_sentences != n_augmentations:
                    repeat = True
                    print('Not all Diseases!')
                    instruction_prompt.append({'role': 'system', 'content': 'Please remember to explicitly mention the disease {} in all the sentences, try again. Just give the answers and nothing else:'.format(disease)})
                else:
                    repeat = False

                if total_reps >= 3:
                    repeat = False
                total_reps += 1
                temperature_aux -= 0.01
                print('')

            # If they are equal
            iterations = 0
            for elem in rephrasing:
                new_disease = np.random.choice(diseases, size=1)[0]
                new_microbe = np.random.choice(microbes, size=1)[0]
                #print(microbe, new_microbe)
                #print(disease, new_disease)
                new_row = copy.deepcopy(row)
                new_row['EVIDENCE'] = elem.strip().lower().replace(microbe.lower(), new_microbe.lower()).replace(disease.lower(), new_disease.lower())
                new_row['DISEASE'] = new_disease
                new_row['MICROBE'] = new_microbe
                new_row['QUESTIONS'] = row['QUESTIONS'].values[0].lower().replace(microbe.lower(), new_microbe.lower()).replace(disease.lower(), new_disease.lower())
                #print(row['QUESTIONS'].values[0].lower().replace(microbe.lower(), new_microbe.lower()).replace(disease.lower(), new_disease.lower()))
                #print(elem.strip().lower().replace(microbe.lower(), new_microbe.lower()).replace(disease.lower(), new_disease.lower()))
                #print('-------')
                new_df.append(new_row)
                iterations += 1

        new_df = pd.concat(new_df, axis=0)
        return new_df



if __name__ == '__main__':
    # LLM
    endpoint_name = 'jumpstart-dft-hf-llm-mixtral-8x7b-instruct'
    model_name = 'mixtral'
    myllm = MyAgent(endpoint_name=endpoint_name, model_name=model_name)

    # template_rephrase = """Given the following excerpt:
    # {sentence}
    #
    # Please rephrase, as scientifically as possible, the previous excerpt clearly stating what is the relationship (if any) between microbe {microbe} and the the disease {disease}.
    # Please don't forget to specifically mention the microbe {microbe} and disease {disease} in your answer"""
    template_rephrase = """Given the following excerpt: 
        {sentence}

        Please rephrase the previous excerpt clearly stating what is the relationship (if any) between microbe {microbe} and the the disease {disease}. 
        Please don't forget to specifically mention the microbe {microbe} and disease {disease} in your answer. Try to keep the scientifical tone and BE BRIEF."""

    template_rephrase = """Given the following excerpt: 
            {sentence}

            Please rephrase the previous excerpt clearly stating what is the relationship (if any) between microbe {microbe} and the the disease {disease}. 
            Please don't forget to specifically mention the microbe {microbe} and disease {disease} in your answer. Try to keep the scientifical tone and BE BRIEF. 
            Give me {n_answers} different rephrases, enumerate them: """

    model_kwargs = {'temperature': 0.5, 'max_new_tokens':500}

    data_augmentator = DataAugmentator(data_path='../../databases/llms-microbe-disease/data/gold_data_corrected.csv',
                                       myllm=myllm)
    data = data_augmentator.data
    relations_counts = data['RELATION'].value_counts().to_dict()
    microbes = np.array(list(set(list(data['MICROBE'].values))))
    diseases = np.array(list(set(list(data['DISEASE'].values))))

    # New augmentations
    llm_augmentations_path = '../normal_db/augmentations/llm_augmentations.csv'
    shuffling_augmentations_path = '../normal_db/augmentations/shuffling_aumentation.csv'
    combine_augmentations_path = '../normal_db/augmentations/shuffling_llm_aumentation.csv'
    llmRAG_augmentions_path = '../normal_db/augmentations/mixtral/'
    data_path = '../../databases/llms-microbe-disease/data/gold_data_corrected.csv'
    data_splits = {}
    for k in range(2, 5):
        print('----------------------------------- k {} -------------------------------'.format(k))
        data_loader = DataLoader(data_path=data_path, use_gold=True, split_gold=True, k=k,
                                 llm_augmentations_path=llm_augmentations_path,
                                 shuffling_augmentations_path=shuffling_augmentations_path,
                                 llmRAG_augmentions_path=llmRAG_augmentions_path,
                                 combine_augmentations_path=combine_augmentations_path)

        data = data_loader.data


        # Just with LLM
        llm_df = data_augmentator.augment_with_llm(data, microbes=microbes, diseases=diseases,
                                                   augmentations_per_class={'positive': 3, 'negative': 4, 'na':6, 'relate':5}, template=template_rephrase,
                                      model_kwargs=model_kwargs)

        llm_df.to_csv('augmentations/llm_augmentations_k{}.csv'.format(k))



    # Just with Shuffling
    # new_negative, new_relate, new_na = data_augmentator.augment_with_shuffling(data, relations_counts)
    # shuffling_df = pd.concat([new_negative, new_relate, new_na], axis=0)
    # shuffling_df.to_csv('augmentations/shuffling_aumentation.csv')

    # Combine Shuffling with LLM
    # llm_augmentations = pd.read_csv('augmentations/llm_augmentations.csv')
    # del llm_augmentations['Unnamed: 0.1']
    # llm_augmentations.index = llm_augmentations['Unnamed: 0']
    # del llm_augmentations['Unnamed: 0']
    #
    # relations_counts_llm = llm_augmentations['RELATION'].value_counts().to_dict()
    # print(relations_counts_llm)
    # new_negative_llm, new_relate_llm, new_na_llm = data_augmentator.augment_with_shuffling(llm_augmentations, relations_counts_llm)
    # shuffling_df = pd.concat([new_negative_llm, new_relate_llm, new_na_llm], axis=0)
    # shuffling_df.to_csv('augmentations/shuffling_llm_aumentation.csv')






