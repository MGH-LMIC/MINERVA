import numpy as np
from openai import OpenAI
import os
import pandas as pd
import copy
import time
import google.generativeai as genai
from sklearn.metrics import accuracy_score

class Reviewer:
    def __init__(self, openai_client, gemini_client):
        self.openai_client = openai_client
        self.gemini_client = gemini_client
        self.answers_dict = {'positive': 0, 'negative': 1, 'na': 2, 'positve':0}

        template_general_instruction = """You are an expert microbiologist who given an excerpt from a research paper can easily 
        identify the type of relation between a microbe and a disease. Doesn't create new information, but is completely faithful to the information provided, and always gives concise answers."""

        template_instruction = """Given the following meaning of the labels, answer the following question with the appropiate label.
        positive: This type is used to annotate microbe-disease entity pairs with positive correlation, such as microbe will cause or aggravate the disease, the microbe will increase when disease occurs.
        negative: This type is used to annotate microbe-disease entity pairs that have a negative correlation, such as microbe can be a treatment for a disease, or microbe will decrease when disease occurs. 
        na: This type is used when the relation between a microbe and a disease is not clear from the context or there is no relation. In other words, use this label if the relation is not positive and not negative."""

        template_evidence = """Based on the above description, evidence is as follows: 
        {evidence}

        "What is the relationship between {microbe} and {disease}? Just Answer with the correct label and nothing else
        """

        template_evidence_choice = """Based on the above description, evidence is as follows: 
                {evidence}

                "What is the relationship between {microbe} and {disease}? In this case you can just pick between {minerva} and {other} labels. Just Answer with the label and nothing else 
                """

        self.template_system = template_general_instruction + '\n' + template_instruction
        self.template_user = template_evidence
        self.template_user_choice = template_evidence_choice

    def openai_predict(self, system_message, user_message):

        completion = self.openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system",
                 "content": system_message},
                {"role": "user",
                 "content": user_message}
                ]
        )

        return completion.choices[0].message.content.strip().lower()

    def gemini_predict(self, system_message, user_message):

        prompt = """ System: {system} \n 
        User: {user} 
        """.format(system=system_message, user=user_message)

        response = self.gemini_client.generate_content(prompt)

        return response.text.strip().lower()

    def review_wrong(self, db_files_wrong, mode='three_alternatives'):
        all_files = db_files_wrong#db_files_correct + db_files_wrong

        for file in all_files:
            db = pd.read_csv(file)
            if mode == 'two_alternatives':
                new_name = file.split('csv')[0] + '_chatgpt_binary.csv'
            else:
                new_name = file.split('csv')[0] + '_chatgpt.csv'
            verifications = []
            print('Analyzing: {}'.format(file))
            for i in range(len(db)):
                if i % 10 == 0 and i != 0:
                    print('{}: {}|{}'.format(file, i, len(db)))
                    pd.concat(verifications).to_csv(new_name)

                row = db.iloc[i]
                evidence = '\n - '.join(row['neo_evidence'].split(' ||| '))
                microbe = row['b_name']
                disease = row['d_name']
                neo_relation = row['neo_relation']
                db_relation = row['db_relation']

                microbe_aux = row['b_name_out']
                disease_aux = row['d_name_out']
                if type(microbe_aux) == str:
                    microbe = microbe_aux
                    disease = disease_aux

                system_message = copy.deepcopy(self.template_system)

                if mode == 'two_alternatives':
                    user_message = copy.deepcopy(self.template_user_choice).format(evidence=evidence, microbe=microbe, disease=disease,
                                                                                   minerva=neo_relation, other=db_relation)
                else:
                    user_message = copy.deepcopy(self.template_user).format(evidence=evidence, microbe=microbe,
                                                                            disease=disease)

                chatgpt_answer = self.openai_predict(system_message, user_message)
                gemini_answer = self.gemini_predict(system_message, user_message)

                time.sleep(0.05)

                row['chatgpt'] = chatgpt_answer
                row['gemini'] = gemini_answer

                verifications.append(row.to_frame('verifications').transpose())

            pd.concat(verifications).to_csv(new_name)

    def calculate_statistics(self, db_files):
        for file in db_files:
            new_name = file.split('csv')[0] + '_chatgpt.csv'
            db = pd.read_csv(new_name)

            neo_relation = np.array([self.answers_dict[elem] for elem in db['neo_relation'].values])
            db_relation = np.array([self.answers_dict[elem] for elem in db['db_relation'].values])
            chatgpt_answer = np.array([self.answers_dict[elem] for elem in db['chatgpt'].values])
            gemini_answer = np.array([self.answers_dict[elem] for elem in db['gemini'].values])

            print('Database: {}'.format(file))
            print('ChatGPT Accuracy MINERVA: {}'.format(round(accuracy_score(chatgpt_answer, neo_relation)*100, 1)))
            print('Gemini Accuracy MINERVA: {}'.format(round(accuracy_score(gemini_answer, neo_relation)*100, 1)))
            print('')
            print('ChatGPT Accuracy other: {}'.format(round(accuracy_score(chatgpt_answer, db_relation)*100, 1)))
            print('Gemini Accuracy other: {}'.format(round(accuracy_score(gemini_answer, db_relation)*100, 1)))
            print('------------------------------------------------------------------')
            print('')




if __name__ == '__main__':
    gemini_key = '[Input Your Gemini key]'
    genai.configure(api_key=gemini_key)

    os.environ['OPENAI_API_KEY'] = '[Input Your OPENAI key]'
    openai_client = OpenAI()

    gemini_client = genai.GenerativeModel('gemini-1.5-pro')
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(m.name)


    db_files_wrong = ['HMDAD/HMDAD_wrong.csv', 'AMADIS/AMADIS_wrong.csv', 'Disbiome/Disbiome_wrong.csv', 'GMDAD/GMDAD_wrong.csv',
                      'Original/Original_wrong.csv']

    db_files_correct = ['AMADIS/AMADIS_correct.csv', 'Disbiome/Disbiome_correct.csv',
                      'GMDAD/GMDAD_correct.csv',
                      'HMDAD/HMDAD_correct.csv', 'Original/Original_correct.csv']


    reviewer = Reviewer(openai_client, gemini_client)

    #reviewer.review_wrong(db_files_wrong, mode='two_alternatives')
    reviewer.calculate_statistics(db_files_wrong)