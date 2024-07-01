import pandas as pd
import numpy as np
import json

class OtherDBsCodifier:
    def __init__(self, databases, diseases_dict, microbes_dict):
        self.databases = databases
        self.databases = pd.concat(self.databases, axis=0, ignore_index=True)
        del self.databases['Unnamed: 0']

        self.diseases_dict = diseases_dict
        self.microbes_dict = microbes_dict

    def process(self):
        self.databases['strength'] = self.databases['neo_strength'].apply(lambda x: np.sign(x))

        self.databases = self.databases[['b_cui', 'd_cui', 'strength']]
        self.databases = self.databases.drop_duplicates()
        self.databases['id_microbe'] = self.databases['b_cui'].map(self.microbes_dict)
        self.databases['id_disease'] = self.databases['d_cui'].map(self.diseases_dict)

        self.databases = self.databases[['id_microbe', 'id_disease', 'strength']]

        return self.databases

if __name__ == '__main__':
    amadis = pd.read_csv('../data/other_dbs/AMADIS/AMADIS_correct_v3.csv')
    gmdad = pd.read_csv('../data/other_dbs/GMDAD/GMDAD_correct_v3.csv')
    hmdad = pd.read_csv('../data/other_dbs/HMDAD/HMDAD_correct_v3.csv')
    disbiome = pd.read_csv('../data/other_dbs/Disbiome/Disbiome_correct_v3.csv')
    original = pd.read_csv('../data/other_dbs/Original/Original_correct_v3.csv')

    with open('../data/diseases_cui_id.json', 'r') as f:
        diseases_dict = json.load(f)

    with open('../data/microbes_cui_id.json', 'r') as f:
        microbes_dict = json.load(f)

    databases = [amadis, gmdad, hmdad,disbiome, original]

    codifier = OtherDBsCodifier(databases, diseases_dict, microbes_dict)
    codifier.process()