import numpy as np
import pandas as pd
import spacy
from scispacy.linking import EntityLinker
from medspacy.ner import TargetRule, TargetMatcher
from collections import defaultdict


class Matcher:
    def __init__(self, path, db_name, folder, diseases_column, bacteria_column, association_column,
                 bacteria_column2='', evidence_column='', paper_column='', associaion_dicts={}):
        if '.xlsx' in path:
            self.db = pd.read_excel(folder + path)
        else:
            self.db = pd.read_csv(folder + path)
        self.db_name = db_name
        self.folder = folder
        self.nlp = spacy.load('en_core_web_sm')
        self.nlp.remove_pipe('ner')
        self.nlp_linker = EntityLinker(nlp=self.nlp, resolve_abbreviations=True, linker_name='umls')
        self.diseases_column = diseases_column
        self.bacteria_column = bacteria_column
        self.bacteria_column2 = bacteria_column2
        self.association_column = association_column
        self.associaion_dicts = associaion_dicts
        self.evidence_column = evidence_column
        self.paper_column = paper_column

    def default_list(self):
        return []

    def parse_bacteria(self, bacteria, prefix='b'):
        self.normalizer_ruler = TargetMatcher(self.nlp)
        new_rules = []
        new_rules += [TargetRule(literal=bacteria, category='BACTERIA')]
        self.normalizer_ruler.add(new_rules)

        # Process bacteria
        doc = self.nlp(bacteria)
        doc = self.normalizer_ruler(doc)
        doc = self.nlp_linker(doc)

        bacteria_dict = {}
        for ent in doc.ents:
            try:
                umls_ent = ent._.kb_ents[0]
                cui = umls_ent[0]
                info = self.nlp_linker.kb.cui_to_entity[cui]
                official_name = info[1]
                bacteria_dict['{}_cui'.format(prefix)] = cui
                bacteria_dict['{}_name'.format(prefix)] = bacteria
                bacteria_dict['{}_official_name'.format(prefix)] = official_name
            except IndexError:
                bacteria_dict['{}_name'.format(prefix)] = bacteria
                bacteria_dict['{}_official_name'.format(prefix)] = None
                bacteria_dict['{}_cui'.format(prefix)] = None

        return bacteria_dict

    def parse_disease(self, disease, prefix='d'):
        self.normalizer_ruler = TargetMatcher(self.nlp)
        new_rules = []
        new_rules += [TargetRule(literal=disease, category='DISEASE')]
        self.normalizer_ruler.add(new_rules)

        # Process bacteria
        doc = self.nlp(disease)
        doc = self.normalizer_ruler(doc)
        doc = self.nlp_linker(doc)

        disease_dict = {}
        for ent in doc.ents:
            try:
                umls_ent = ent._.kb_ents[0]
                cui = umls_ent[0]
                info = self.nlp_linker.kb.cui_to_entity[cui]
                official_name = info[1]
                disease_dict['{}_cui'.format(prefix)] = cui
                disease_dict['{}_name'.format(prefix)] = disease
                disease_dict['{}_official_name'.format(prefix)] = official_name
            except IndexError:
                disease_dict['{}_name'.format(prefix)] = disease
                disease_dict['{}_official_name'.format(prefix)] = None
                disease_dict['{}_cui'.format(prefix)] = None

        if len(doc.ents) == 0:
            disease_dict['{}_name'.format(prefix)] = disease
            disease_dict['{}_official_name'.format(prefix)] = None
            disease_dict['{}_cui'.format(prefix)] = None

        return disease_dict


    def parse_association(self, relation):
        relation = str(relation)
        try:
            relation = self.associaion_dicts[relation]
        except:
            print(relation)
            if 'positive corr' in relation.lower():
                relation = 'positive'
            elif 'negative corr' in relation.lower():
                relation = 'negative'
            else:
                print('Unknown relation: {}'.format(relation))
                relation = 'na'

        return relation

    def match_connections(self):
        out_dicto = defaultdict(self.default_list)
        for i in range(len(self.db)):
            row = self.db.iloc[i]
            if i % 100 == 0:
                print('{}/{}'.format(i, len(self.db)))

            # Processing Bacterias
            bacteria = self.parse_bacteria(row[self.bacteria_column], prefix='b')
            for k, v in bacteria.items():
                out_dicto[k].append(v)

            if self.bacteria_column2 != '':
                bacteria2 = self.parse_bacteria(row[self.bacteria_column], prefix='b2')
                for k, v in bacteria2.items():
                    out_dicto[k].append(v)

            # Processing diseases
            disease = self.parse_disease(row[self.diseases_column], prefix='d')
            for k, v in disease.items():
                out_dicto[k].append(v)

            # Processing relations
            relation = self.parse_association(row[self.association_column])
            out_dicto['relation'].append(relation)
            out_dicto['original_relation'].append(row[self.association_column])

            # Processing evidence and paper
            if self.evidence_column != '':
                out_dicto['evidence'].append(row[self.evidence_column])
            else:
                out_dicto['evidence'].append('')

            # Processing evidence and paper
            if self.paper_column != '':
                out_dicto['paper'].append(row[self.paper_column])
            else:
                out_dicto['paper'].append('')


        out_dicto = pd.DataFrame(out_dicto)
        out_dicto.to_csv(self.folder + '{}_new.csv'.format(self.db_name))


if __name__ == '__main__':
    db_name = 'AMADIS'

    ########################################## AMADIS ################################################
    if db_name == 'AMADIS':
        path = 'GIFTED.xlsx'
        folder = 'AMADIS/'
        diseases_columns = 'Disease'
        bacteria_column = 'Flora'
        bacteria_column2 = 'Phylum'
        association_column = 'Association between disease and microflora'
        evidence_column = ''
        paper_column = 'PubmedID'
        associaion_dicts = {"Associated": 'na', "Microflora promote disease's progression": "positive",
                            "Positive correlation": "positive", "Negative correlation": "negative", "Unassociated": "na",
                            "Microflora inhibit disease's progression": "negative", 'NA': 'na'}


        matcher = Matcher(path, db_name=db_name, folder=folder, diseases_column=diseases_columns,
                          evidence_column=evidence_column, paper_column=paper_column,
                          bacteria_column=bacteria_column, association_column=association_column,
                          bacteria_column2=bacteria_column2, associaion_dicts=associaion_dicts)

        matcher.match_connections()

    elif db_name == 'GMDAD':
        path = 'GMMAD.csv'
        folder = 'GMDAD/'
        diseases_columns = 'DISEASE'
        bacteria_column = 'BACTERIA'
        bacteria_column2 = ''
        association_column = 'RELATION'
        evidence_column = ''
        paper_column = 'PMID'
        associaion_dicts = {"Increase": "positive", "Decrease": "negative"}

        matcher = Matcher(path, db_name=db_name, folder=folder, diseases_column=diseases_columns,
                          evidence_column=evidence_column, paper_column=paper_column,
                          bacteria_column=bacteria_column, association_column=association_column,
                          bacteria_column2=bacteria_column2, associaion_dicts=associaion_dicts)

        matcher.match_connections()

    elif db_name=='HMDAD':
        path = 'hmdad.csv'
        folder = 'HMDAD/'
        diseases_columns = 'Disease'
        bacteria_column = 'Microbe'
        bacteria_column2 = ''
        association_column = 'Evidence'
        evidence_column = ''
        paper_column = 'PMID'
        associaion_dicts = {"Increase": "positive", "Decrease": "negative"}

        matcher = Matcher(path, db_name=db_name, folder=folder, diseases_column=diseases_columns,
                          evidence_column=evidence_column, paper_column=paper_column,
                          bacteria_column=bacteria_column, association_column=association_column,
                          bacteria_column2=bacteria_column2, associaion_dicts=associaion_dicts)

        matcher.match_connections()

    elif db_name == 'original':
        path = 'gold_data_corrected.csv'
        folder = 'Original/'
        diseases_columns = 'DISEASE'
        bacteria_column = 'MICROBE'
        bacteria_column2 = ''
        association_column = 'RELATION'
        evidence_column = 'EVIDENCE'
        paper_column = ''
        associaion_dicts = {"positive": "positive", "negative": "negative", "relate": "na", "na": "na"}

        matcher = Matcher(path, db_name=db_name, folder=folder, diseases_column=diseases_columns,
                          evidence_column=evidence_column, paper_column=paper_column,
                          bacteria_column=bacteria_column, association_column=association_column,
                          bacteria_column2=bacteria_column2, associaion_dicts=associaion_dicts)

        matcher.match_connections()

    else:
        path = 'disbiome.csv'
        folder = 'Disbiome/'
        diseases_columns = 'disease_name'
        bacteria_column = 'organism_name'
        bacteria_column2 = ''
        association_column = 'qualitative_outcome'
        evidence_column = 'meddra_id'
        paper_column = ''
        associaion_dicts = {"Elevated": "positive", "Reduced": "negative"}

        matcher = Matcher(path, db_name=db_name, folder=folder, diseases_column=diseases_columns,
                          evidence_column=evidence_column, paper_column=paper_column,
                          bacteria_column=bacteria_column, association_column=association_column,
                          bacteria_column2=bacteria_column2, associaion_dicts=associaion_dicts)

        matcher.match_connections()
