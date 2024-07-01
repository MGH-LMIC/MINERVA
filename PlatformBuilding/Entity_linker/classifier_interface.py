from pymongo import MongoClient
import json
from collections import defaultdict
from collections import Counter
from graph_creator import GraphCreator
import numpy as np
from llm_classifier.llm_classifier import LLMClassifier
from llm_classifier.my_agent import MyAgent
import time


def defaultlist():
    return []

class ClassifierInterface:
    def __init__(self, model, mongo_db='microbiome_abstracts_db', mongo_collection_source_name='microbiome_full_papers_sentences',
                 mongo_collection_target_name='microbiome_classification'):

        self.model = model
        self.model_name = model.model_name
        self.mongo_collection_source_name = mongo_collection_source_name
        self.mongo_collection_target_name = mongo_collection_target_name + '_{}'.format(self.model_name)
        self.mongo_collection_target_other_name = mongo_collection_target_name + '_{}_other'.format(self.model_name)
        self.mongo_client = MongoClient('localhost', 27017)
        self.mongo_collection_source = self.mongo_client[mongo_db][self.mongo_collection_source_name]
        self.mongo_collection_target = self.mongo_client[mongo_db][self.mongo_collection_target_name]
        self.mongo_collection_target_other = self.mongo_client[mongo_db][self.mongo_collection_target_other_name]
        self.all_ids = self.mongo_collection_source.distinct('_id')
        self.save_ids = self.mongo_collection_target.distinct('paper_id')


        # Make the intersection
        input_ids = set([elem.split('_')[0] for elem in self.all_ids])
        print('Papers total: {}'.format(len(input_ids)))
        print('Papers already analyzed: {}'.format(len(self.save_ids)))
        output_ids = set(self.save_ids)
        self.papers_ids_to_analyze = list(input_ids - output_ids)
        print('Papers to Analyze: {}'.format(len(self.papers_ids_to_analyze)))




        # Graph Creator
        self.graph_creator = GraphCreator()

        # Trying to rescue the index
        try:
            with open('paper_index_classifier.json', 'r') as f:
                dicto_save = json.load(f)
                self.index = dicto_save['paper_index']
                self.total_predicted_relations = dicto_save['total_predicted_relations']

        except FileNotFoundError:
            print('Index Not Found')
            self.index = 0
            self.total_predicted_relations = 0

    def group_ids(self):
        all_ids = self.all_ids
        dicto_ids = defaultdict(defaultlist)
        for elem in all_ids:
            paper_id = elem.split('_')[0]
            dicto_ids[paper_id].append(elem)
        return dicto_ids

    def save_to_mongo(self, paper_info, disease_info, bacteria_info, evidence, relation):
        to_db = {}
        to_db.update(paper_info)
        to_db['evidence'] = evidence
        to_db['relation'] = relation
        to_db['disease'] = disease_info
        to_db['bacteria_info'] = bacteria_info

        self.mongo_collection_target.insert_one(to_db)


    def save_to_mongo_other(self, paper_info, disease_info, bacteria_info, evidence, relation):
        to_db = {}
        to_db['paper_inf'] = paper_info
        to_db['evidence'] = evidence
        to_db['relation'] = relation
        to_db['disease_info'] = disease_info
        to_db['bacteria_info'] = bacteria_info

        self.mongo_collection_target_other.insert_one(to_db)



    def run(self):
        db_length = self.mongo_collection_source.count_documents({})
        dicto_ids = self.group_ids()
        for paper_id in self.papers_ids_to_analyze:
            sentence_ids = dicto_ids[paper_id]
            print('Classifying Sentences of Paper {} ({}|{}) | N°Sentences {}'.format(paper_id, self.index + 1, db_length, len(sentence_ids)))

            # Paper information
            paper = self.mongo_collection_source.find_one({'_id': sentence_ids[0]})
            paper_dicto = {'title':  paper['title'], 'publication_date': paper['publication_date'],
                           'issn': paper['issn'], 'journal': paper['journal'], 'pubmed_id': paper['pubmed_id'],
                           'pmc_id': paper['pmc_id'], 'paper_num': paper['paper_num'], 'paper_id': paper_id}


            # Going through each of the sentences
            dicto_relations = defaultdict(defaultlist)
            dicto_diseases = {}
            dicto_bacterias = {}
            for sentence_id in sentence_ids:
                paper = self.mongo_collection_source.find_one({'_id': sentence_id})

                # Getting the sentence and all found bacterias and diseases
                evidence = paper['sentence']

                if len(evidence.split(' ')) >= 500:
                    continue
                if len(evidence) >= 5000:
                    continue

                diseases = paper['diseases']
                bacterias = paper['bacterias']
                for d in diseases:
                    for b in bacterias:
                        disease = d['disease']
                        bacteria = b['bacteria']

                        disease_cui = d['cui']
                        bacteria_cui = b['cui']

                        if bacteria.lower() == 'bacteria' or bacteria.lower() == 'bacterias':
                            continue

                        if bacteria.lower() == 'probiotic' or bacteria.lower() == 'probiotics':
                            continue

                        if disease.lower() == 'disease' or disease.lower() == 'diseases':
                            continue

                        # Perform Prediction
                        relation = self.model.run(sentence=evidence, disease=disease, microbe=bacteria,
                                                  model_kwargs=[{'temperature': 0.7, 'max_new_tokens': 20, 'do_sample': True} for _ in range(7)],
                                                  zero_shot=True, rephrase=False, n_rephrasings=1)


                        print('{} predicted relation: {} {}'.format('#'*29, relation, '#'*29))

                        if relation == 'na':
                            continue

                        # Saving all in the dictionary to perform the paper voting later
                        dicto_relations[(disease_cui, bacteria_cui)].append({'relation': relation, 'evidence': evidence})

                        dicto_diseases[disease_cui] = {'name': disease.lower(),
                             'definition': d['definition'],
                             'official_name': d['official_name'],
                             'synonyms': d['synonyms'][:5], 'cui': disease_cui, 'tui': d['tui']}

                        dicto_bacterias[bacteria_cui] = {'name': bacteria.lower(),
                                                       'definition': b['definition'],
                                                       'official_name': b['official_name'],
                                                       'synonyms': b['synonyms'][:5], 'cui': bacteria_cui, 'tui': b['tui']}

                        self.total_predicted_relations += 1


            # Summarizing paper information
            if len(dicto_relations) != 0:
                for key, relations in dicto_relations.items():
                    evidences = []
                    if len(relations) == 1: # Not discussion
                        evidences.append(relations[0]['evidence'])
                        final_relation = relations[0]['relation']
                    else:
                        # Only Keeping the most common
                        relations_counts = Counter(r['relation'] for r in relations)
                        most_common = relations_counts.most_common(1)[0][0]
                        for i in range(len(relations)):
                            if relations[i]['relation'] == most_common:
                                evidences.append(relations[i]['evidence'])

                        evidences = list(set(evidences))
                        final_relation = most_common

                    # Now saving into the db
                    evidences = ' ||| '.join(evidences)
                    disease_cui = key[0]
                    bacteria_cui = key[1]

                    disease_info = dicto_diseases[disease_cui]
                    bacteria_info = dicto_bacterias[bacteria_cui]

                    self.save_to_mongo(paper_info=paper_dicto, disease_info=disease_info, bacteria_info=bacteria_info,
                                      evidence=evidences, relation=final_relation)


                    # Creating Graph form
                    self.graph_creator.create_relation(paper_info=paper_dicto, disease_info=disease_info,
                                                       bacteria_info=bacteria_info, evidence=evidences, relation=final_relation)
            else:
                self.save_to_mongo(paper_info=paper_dicto, disease_info={}, bacteria_info={},
                                   evidence='', relation='')


            self.index += 1
            dicto_save = {'paper_index': self.index, 'total_predicted_relations': self.total_predicted_relations}
            with open('paper_index_classifier.json', 'w') as f:
                json.dump(dicto_save, f)


class Classifier_aux:
    def __init__(self):
        self.model_name = 'test_model'

    def predict(self, evidence, disease, bacteria):
        return np.random.choice(['positive', 'negative'])


if __name__ == '__main__':
    endpoint = 'huggingface-pytorch-tgi-inference-2024-05-02-16-40-28-306'
    model_name = 'biomistral'

    llm = MyAgent(endpoint_name=endpoint, model_name=model_name)
    model = LLMClassifier(llm=llm, model_name=model_name)

    # Classification
    classifier = ClassifierInterface(model=model, mongo_db='microbiome_abstracts_db', mongo_collection_source_name='microbiome_sentences',
                 mongo_collection_target_name='microbiome_relations')
    classifier.run()