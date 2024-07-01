import pandas as pd
from pymongo import MongoClient
from spacy.pipeline import Sentencizer
import spacy
import re
import torch
import pymongo
from scispacy.linking import EntityLinker
from transformers import AutoTokenizer, AutoModelForTokenClassification
from bacteria_ner_utils import get_prediction, DistilbertNER
from transformers import pipeline
from scispacy.linking import EntityLinker
from scispacy.abbreviation import AbbreviationDetector
from medspacy.ner import TargetRule, TargetMatcher
import json


def replace_decimal(match):
    return f"{match.group(1)}_{match.group(2)}"

def remove_short_strings_in_parenthesis(text):
    # Define a regular expression pattern to find strings inside parenthesis
    pattern = r'\([^)]{1,3}\)'  # Matches strings inside parenthesis that are 1 to 3 characters long

    # Use re.sub() to replace matches with an empty string
    result = re.sub(pattern, '', text)

    return result

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def find_substring_indices(string, substring):
    start_index = string.find(substring)
    if start_index == -1:
        return None  # Substring not found
    end_index = start_index + len(substring) - 1
    return [start_index, end_index]



class ProcessPapers:
    def __init__(self, mongo_db='microbiome_db', mongo_collection_source='microbiome',
                 mongo_collection_target='microbiome_sentences'):
        self.mongo_collection_source_name = mongo_collection_source

        self.mongo_client = MongoClient('localhost', 2717)
        self.mongo_collection = self.mongo_client[mongo_db][mongo_collection_source]
        self.mongo_collection_target = self.mongo_client[mongo_db][mongo_collection_target]

        inserted_papers = self.mongo_collection_target.distinct('pubmed_id')
        if 'full_paper' in self.mongo_collection_source_name:
            source_papers = self.mongo_collection.distinct('pubmed_id')
        else:
            source_papers = self.mongo_collection.distinct('_id')

        self.papers_to_insert = set(source_papers) - set(inserted_papers)
        self.papers_to_insert = list(self.papers_to_insert)


        self.nlp = spacy.load("en_core_sci_sm")

        self.sentence_matcher = Sentencizer(punct_chars=['.', '!', '?', '*', '\n'])

        # Diseases NER
        self.disease_model = spacy.load('en_ner_bc5cdr_md')

        # Bacteria NERs
        self.bacteria_model_spacy_bert = spacy.load('en_core_sci_lg')
        self.bacteria_model_spacy_bert.add_pipe("abbreviation_detector")
        self.bacteria_model_spacy_bert.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "mesh"})

        model_path = 'bacteria_distillBERT_silver_20_epochs_B_large.pt'#'bacteria_distillBERT_silver_20_epochs_B_2.pt'
        self.bacteria_ner = torch.load(model_path, map_location='cpu').to(device)
        self.bacteria_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

        # Normalization
        self.normalizer = spacy.load("en_core_sci_lg")
        self.normalizer.remove_pipe('ner')
        self.normalizer_linker = EntityLinker(nlp=self.normalizer, resolve_abbreviations=True, linker_name='umls')
        self.dicto_definitions = {}

        # Trying to rescue the index
        try:
            if 'full_paper' in self.mongo_collection_source_name:
                with open('paper_index_full.json', 'r') as f:
                    dicto_save = json.load(f)
                    self.index = 0
                    self.inserted_sentences = dicto_save['inserted_sentences']
                    self.total_sentences = dicto_save['total_sentences']
            else:
                with open('paper_index.json', 'r') as f:
                    dicto_save = json.load(f)
                    self.index = 0
                    self.inserted_sentences = dicto_save['inserted_sentences']
                    self.total_sentences = dicto_save['total_sentences']

        except FileNotFoundError:
            print('Index Not Found')
            self.index = 0
            self.inserted_sentences = 0
            self.total_sentences = 0

    def split_sentences(self, abstract):
        # Split into sections
        abstract = abstract[:1000000]
        doc = self.nlp(abstract.lower())


        # Pattern for decimal numbers not splitted
        pattern = r'(\d+)\.(\d+)'
        sentences = self.sentence_matcher(doc).sents
        all_sentences = []
        for sent in sentences:
            sent = re.sub(pattern, replace_decimal, sent.text.strip())

            #sent = (sent.replace('?', '.').replace('!', '.').replace('\n', '.')
            #            .replace(';', '.').split('.'))

            #sent = [elem for elem in sent if
            #        len(elem) > 1 and len(elem.split(' ')) > 1]  # More than one letter and more than one word
            all_sentences += [sent]

        all_sentences = [sentence.strip().lower() for sentence in all_sentences if len(sentence.strip()) > 1]

        all_sentences = [sentence.replace('-', ' ').replace('/', ' ').replace('|', ' ')
                                 for sentence in all_sentences if len(sentence.strip()) > 1]

        # Solve abbreviations
        all_sentences_aux = []
        for sentence in all_sentences:
            doc = self.bacteria_model_spacy_bert(sentence)
            for abrv in doc._.abbreviations:
                sentence = sentence.replace(abrv.text, abrv._.long_form.text)
            if len(sentence.split(' ')) > 100:
                sentences = sentence.split(' ')
                total_len = len(sentences)
                index = 0
                while index < total_len:
                    all_sentences_aux.append(' '.join(sentences[index: index + 100]))

                    index += 70
            else:
                all_sentences_aux.append(sentence)

        all_sentences = all_sentences_aux

        return all_sentences


    def extract_diseases(self, all_sentences):
        sentences_with_diseases = []
        for sentence in all_sentences:
            # NER2
            doc = self.disease_model(sentence)
            ner_diseases = []
            for ent in doc.ents:
                if ent.label_ == 'DISEASE':
                    ner_diseases.append(ent.text)

            if ner_diseases != []:
                sentences_with_diseases.append({'sentence': sentence, 'diseases': list(set(ner_diseases))})

        return sentences_with_diseases


    def extract_bacterias(self, filtered_sentences):
        if filtered_sentences == []:
            return []

        new_filtered_sentences = []
        for elem in filtered_sentences:
            diseases = elem['diseases']
            sentence = elem['sentence']

            # Spacy model
            doc = self.bacteria_model_spacy_bert(sentence)
            linker = self.bacteria_model_spacy_bert.get_pipe("scispacy_linker")
            bacterias_ner1 = []
            for ent in doc.ents:
                try:
                    umls_ent = ent._.kb_ents[0]
                except IndexError:
                    continue
                cui = umls_ent[0]
                similarity = umls_ent[1] # TUI(s)
                info = linker.kb.cui_to_entity[cui]
                tui = info[3]
                if similarity > 0.9 and tui[0] == 'T007':
                    bacterias_ner1.append(ent.text)

            # Adham's model
            prediction = get_prediction(model=self.bacteria_ner, tokenizer=self.bacteria_tokenizer, sentence=sentence, NER='B')['B']
            bacterias_ner2 = []
            if prediction != []:
                aux_sentence = sentence.split(' ')
                aux_sentence = [elem.replace('(', '').replace(')', '').replace('.', '').replace(',', '').replace(';', '') for elem in aux_sentence]
                for i in range(len(aux_sentence)):
                    if aux_sentence[i] in prediction:

                        if i < len(aux_sentence) - 1:
                            if aux_sentence[i + 1] in prediction:
                                #print('in', aux_sentence[i + 1])
                                bacterias_ner2.append('{} {}'.format(aux_sentence[i], aux_sentence[i + 1]))
                            else:
                                #print('not in', aux_sentence[i + 1])
                                if len(bacterias_ner2) > 0:
                                    if aux_sentence[i] not in bacterias_ner2[-1]:
                                        bacterias_ner2.append(aux_sentence[i])
                                else:
                                    bacterias_ner2.append(aux_sentence[i])

                        else:
                            bacterias_ner2.append(aux_sentence[i])

            bacterias_ner = bacterias_ner2 + bacterias_ner1
            bacterias_ner = list(set(bacterias_ner))

            if bacterias_ner != []:
                new_filtered_sentences.append({'sentence': sentence, 'diseases': diseases, 'bacterias': bacterias_ner})
        return new_filtered_sentences

    def eliminate_confusions(self, filtered_sentences):
        if filtered_sentences == []:
            return []
        else:
            new_filtered_sentences = []
            for elem in filtered_sentences:
                diseases_to_eliminate = []
                bacterias_to_eliminate = []

                # Finding string ranges to look for intersections
                diseases_indices = [find_substring_indices(elem['sentence'], disease) for disease in elem['diseases']]
                bacterias_indices = [find_substring_indices(elem['sentence'], bacteria) for bacteria in elem['bacterias']]
               
                for i in range(len(diseases_indices)):
                    for j in range(len(bacterias_indices)):
                        if diseases_indices[i] != None and bacterias_indices[j] != None:
                            diseases_range = range(diseases_indices[i][0], diseases_indices[i][1])
                            bacteria_range = range(bacterias_indices[j][0], bacterias_indices[j][1])
                            intersection = list(set(diseases_range).intersection(bacteria_range))
                            if intersection != []:
                                if len(diseases_range) > len(bacteria_range): # Bacteria is contained in disease
                                    bacterias_to_eliminate.append(elem['bacterias'][j])
                                else:
                                    diseases_to_eliminate.append(elem['diseases'][i])

                diseases_to_eliminate = list(set(diseases_to_eliminate))
                #bacterias_to_eliminate = list(set(bacterias_to_eliminate))

                # Eliminating those which were not able to be found in the range search
                diseases = [elem['diseases'][i] for i in range(len(elem['diseases'])) if diseases_indices[i] != None]
                bacterias = [elem['bacterias'][i] for i in range(len(elem['bacterias'])) if bacterias_indices[i] != None]

                # Eliminiating the interesections
                diseases = [disease for disease in diseases if disease not in diseases_to_eliminate]
                bacterias = [bacteria for bacteria in bacterias if bacteria not in bacterias_to_eliminate]

                if diseases != [] and bacterias != []:
                    new_filtered_sentences.append({'sentence': elem['sentence'], 'diseases': diseases, 'bacterias': bacterias})

            return new_filtered_sentences



    def normalization(self, filtered_sentences):
        if filtered_sentences == []:
            return []

        new_filtered_sentences = []
        for elem in filtered_sentences:
            sentence = elem['sentence']
            diseases = list(set(elem['diseases']))
            bacterias = list(set(elem['bacterias']))

            # Create new rules
            self.normalizer_ruler = TargetMatcher(self.normalizer)
            new_rules = []
            new_rules += [TargetRule(literal=disease, category='DISEASE') for disease in diseases]
            new_rules += [TargetRule(literal=bacteria, category='BACTERIA') for bacteria in bacterias]
            self.normalizer_ruler.add(new_rules)

            # Catching Entities and definitions
            doc = self.normalizer(sentence)
            doc = self.normalizer_ruler(doc)
            doc = self.normalizer_linker(doc)

            new_diseases = []
            new_bacterias = []
            for ent in doc.ents:
                label = ent.label_
                if label == 'BACTERIA':
                    try:
                        umls_ent = ent._.kb_ents[0]
                        cui = umls_ent[0]
                        similarity = umls_ent[1]  # TUI(s)
                        info = self.normalizer_linker.kb.cui_to_entity[cui]
                        tui = info[3]
                        definition = info[4]
                        if tui[0] == 'T007': # Is a bacteria
                            new_bacterias.append({'bacteria': ent.text, 'cui': cui, 'tui': tui[0],
                                                  'definition': definition, 'similarity':similarity,
                                                      'official_name': info[1], 'synonyms': info[2]})
                    except IndexError:
                        pass
                elif label == 'DISEASE':
                    try:
                        umls_ent = ent._.kb_ents[0]
                        cui = umls_ent[0]
                        similarity = umls_ent[1]  # TUI(s)
                        info = self.normalizer_linker.kb.cui_to_entity[cui]
                        tui = info[3]
                        definition = info[4]
                        if tui[0] in ['T019', 'T020', 'T033', 'T037', 'T046', 'T047', 'T048', 'T049', 'T184', 'T190', 'T191']:
                            new_diseases.append({'disease': ent.text, 'cui': cui, 'tui': tui[0], 'definition': definition,
                                                 'similarity': similarity, 'official_name': info[1], 'synonyms': info[2]})
                    except IndexError:
                        pass
                else:
                    pass


            # Finally if one of them is empty, we just ommit
            if new_diseases != [] and new_bacterias != []:
                new_filtered_sentences.append({'sentence': sentence, 'disease': new_diseases, 'bacterias': new_bacterias})

        return new_filtered_sentences


    def run(self):
        inserted_sentences = self.inserted_sentences
        len_papers_to_insert = len(self.papers_to_insert)

        for pubmed_id in self.papers_to_insert:
            if 'full_paper' in self.mongo_collection_source_name:
                paper = self.mongo_collection.find_one({'pubmed_id': pubmed_id})
            else:
                paper = self.mongo_collection.find_one({'_id': pubmed_id})

            print('Analyzing Paper: ({}|{}) {}'.format(len_papers_to_insert, self.index, pubmed_id))
            print('Inserted sentences total: {}'.format(inserted_sentences))
            print('-------------------------------------------------------------------------------')
            publication_date = paper['publication_date']
            title = paper['title']
            journal = paper['journal']
            issn = paper['issn']
            if 'full_paper' in self.mongo_collection_source_name:
                all_text = paper['full_text']
            else:
                abstract = paper['abstract']
                all_text = abstract

            # Splitting by sentence
            all_sentences = self.split_sentences(all_text)
            self.total_sentences += len(all_sentences)

            # Extract diseases
            sentences_with_diseases = self.extract_diseases(all_sentences)

            # Extract Bacterias
            sentences_with_bacterias_and_diseases = self.extract_bacterias(sentences_with_diseases)

            # Eliminating sentences in which the found disease is equal or is contained in the found bacteria
            filtered_sentences = self.eliminate_confusions(sentences_with_bacterias_and_diseases)

            # Normalization of found entities
            filtered_sentences = self.normalization(filtered_sentences)

            # To db
            self.index += 1
            if filtered_sentences != []:
                sentence_idx = 0
                for sentence in filtered_sentences:
                    if 'full_paper' in self.mongo_collection_source_name:
                        sentence_dict = {'title': title, 'publication_date': str(publication_date), 'issn': issn,
                                         'all_text': '', '_id': '{}_n{}'.format(pubmed_id, sentence_idx),
                                         'sentence': sentence['sentence'], 'journal': journal,
                                         'diseases': sentence['disease'],
                                         'bacterias': sentence['bacterias'], 'paper_num': 0,
                                         'pubmed_id': paper['pubmed_id'], 'pmc_id': str(paper['_id'])}
                    else:
                        sentence_dict = {'title': title, 'publication_date': str(publication_date), 'issn': issn,
                                         'all_text': all_text, '_id': '{}_n{}'.format(pubmed_id, sentence_idx),
                                         'sentence': sentence['sentence'], 'journal': journal,
                                         'diseases': sentence['disease'],
                                         'bacterias': sentence['bacterias'], 'paper_num': 0,
                                         'pubmed_id': str(paper['_id']), 'pmc_id': ''}

                    try:
                        self.mongo_collection_target.insert_one(sentence_dict)
                        inserted_sentences += 1
                        sentence_idx += 1
                    except pymongo.errors.DuplicateKeyError as e:
                        print(e)
                        print('---')

            else:
                if 'full_paper' in self.mongo_collection_source_name:
                    sentence_dict = {'title': title, 'publication_date': str(publication_date), 'issn': issn,
                                     'all_text': '', '_id': '{}_n{}'.format(pubmed_id, 0),
                                     'sentence': '', 'journal': journal,
                                     'diseases': '',
                                     'bacterias': '', 'paper_num': 0,
                                     'pubmed_id': paper['pubmed_id'], 'pmc_id': str(paper['_id'])}
                else:
                    sentence_dict = {'title': title, 'publication_date': str(publication_date), 'issn': issn,
                                     'all_text': '', '_id': '{}_n{}'.format(pubmed_id, 0),
                                     'sentence': '', 'journal': journal,
                                     'diseases': '',
                                     'bacterias': '', 'paper_num': 0,
                                     'pubmed_id': str(paper['_id']), 'pmc_id': ''}
                try:
                    self.mongo_collection_target.insert_one(sentence_dict)
                except pymongo.errors.DuplicateKeyError as e:
                    print(e)
                    print('---')



            # Saving index
            if 'full_paper' in self.mongo_collection_source_name:
                with open('paper_index_full.json', 'w') as f:
                    json.dump({'papers_index': self.index, 'inserted_sentences': inserted_sentences,
                               'total_sentences': self.total_sentences}, f)
            else:
                with open('paper_index.json', 'w') as f:
                    json.dump({'papers_index': self.index, 'inserted_sentences': inserted_sentences,
                               'total_sentences': self.total_sentences}, f)



if __name__ == '__main__':
    process_type = 'abstracts'

    if process_type == 'abstracts':
        process_papers = ProcessPapers(mongo_db='microbiome_abstracts_db', mongo_collection_source='microbiome_abstracts',
                                   mongo_collection_target='microbiome_abstracts_sentences')
    else:
        process_papers = ProcessPapers(mongo_db='microbiome_abstracts_db', mongo_collection_source='microbiome_abstracts',
                                   mongo_collection_target='microbiome_sentences')
    process_papers.run()