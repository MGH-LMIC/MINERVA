from py2neo import Graph, NodeMatcher
import pandas as pd
import os
import copy
import json

class DataGetter:
    def __init__(self, data_folder='data/'):
        self.graph = Graph(host="0.0.0.0", port="7687", auth=('neo4j', 'your_password'))
        self.data_folder = data_folder
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)

    def get_microbes(self):
        query = """
        Match (n:Microbe) Return n.cui as cui, n.name as name, n.official_name as official_name, n.definition as definition
        """
        out = self.graph.run(query).to_data_frame()
        out.index.name = 'id'
        return out

    def get_diseases(self):
        query = """
        Match (n:Disease) Return n.cui as cui, n.name as name, n.official_name as official_name, n.definition as definition
        """
        out = self.graph.run(query).to_data_frame()
        out.index.name = 'id'
        return out

    def get_parent_relationships(self):
        query = """
        Match (n:Microbe)-[r:PARENT]-(m:Microbe) Return n.cui as cui_orig, m.cui as cui_target, n.name as name_orig, m.name as name_target 
        """
        out_microbe = self.graph.run(query).to_data_frame()
        out_microbe.index.name = 'id'

        query = """
                Match (n:Disease)-[r:PARENT]-(m:Disease) Return n.cui as cui_orig, m.cui as cui_target, n.name as name_orig, m.name as name_target 
                """
        out_disease = self.graph.run(query).to_data_frame()
        out_disease.index.name = 'id'

        return out_microbe, out_disease


    def get_strength_relations(self):
        query = """
                Match (n:Microbe)-[r:STRENGTH]-(m:Disease) Return n.cui as cui_microbe, m.cui as cui_disease, r.strength_raw as strength_raw, r.strength_IF as strength_IF
                """
        out = self.graph.run(query).to_data_frame()
        out.index.name = 'id'
        return out

    def get_negative_positive_relations(self):
        query = """
        MATCH (m:Microbe)-[s:STRENGTH]-(d:Disease)
            WITH m, d, s
            OPTIONAL MATCH (m)-[neg:NEGATIVE]-(d)
            WITH m, d, s, COUNT(DISTINCT neg) AS neg_relations
            OPTIONAL MATCH (m)-[pos:POSITIVE]-(d)
            RETURN 
                m.cui AS cui_microbe,
                d.cui AS cui_disease,
                neg_relations,
                COUNT(DISTINCT pos) AS pos_relations,
                s.strength_raw AS strength_raw,
                s.strength_IF as strength_IF
        """
        out = self.graph.run(query).to_data_frame()
        out.index.name = 'id'
        return out

    def get_all_data_from_neo4j(self, save_csv=True):

        # Get all the data
        microbes = data_getter.get_microbes()
        diseases = data_getter.get_diseases()
        microbes_genealogy, diseases_genealogy = data_getter.get_parent_relationships()
        strenghts = data_getter.get_strength_relations()

        neg_pos_strength = data_getter.get_negative_positive_relations()

        # Create a dictionary with all the cuis for microbes and diseases
        microbes_id_cui = microbes['cui'].to_dict()
        microbes_cui_id = {v:k for k,v in microbes_id_cui.items()}

        diseases_id_cui = diseases['cui'].to_dict()
        disease_cui_id = {v: k for k, v in diseases_id_cui.items()}

        # Changing the original dataframes to include the ids
        
        # Parent relationships first
        microbes_genealogy_with_id = copy.deepcopy(microbes_genealogy)
        microbes_genealogy_with_id['cui_orig'] = microbes_genealogy_with_id['cui_orig'].map(microbes_cui_id)
        microbes_genealogy_with_id['cui_target'] = microbes_genealogy_with_id['cui_target'].map(microbes_cui_id)
        microbes_genealogy_with_id = microbes_genealogy_with_id.rename({'cui_orig': 'id_orig', 'cui_target': 'id_target'}, axis=1)
        
        
        diseases_genealogy_with_id = copy.deepcopy(diseases_genealogy)
        diseases_genealogy_with_id['cui_orig'] = diseases_genealogy_with_id['cui_orig'].map(disease_cui_id)
        diseases_genealogy_with_id['cui_target'] = diseases_genealogy_with_id['cui_target'].map(disease_cui_id)
        diseases_genealogy_with_id = diseases_genealogy_with_id.rename({'cui_orig': 'id_orig', 'cui_target': 'id_target'}, axis=1)

        # Strength relationship
        strenghts_with_id = copy.deepcopy(strenghts)
        strenghts_with_id['cui_microbe'] = strenghts_with_id['cui_microbe'].map(microbes_cui_id)
        strenghts_with_id['cui_disease'] = strenghts_with_id['cui_disease'].map(disease_cui_id)
        strenghts_with_id = strenghts_with_id.rename({'cui_microbe': 'id_microbe', 'cui_disease': 'id_disease'}, axis=1)

        # All relationships
        neg_pos_strength_with_id = copy.deepcopy(neg_pos_strength)
        neg_pos_strength_with_id['cui_microbe'] = neg_pos_strength_with_id['cui_microbe'].map(microbes_cui_id)
        neg_pos_strength_with_id['cui_disease'] = neg_pos_strength_with_id['cui_disease'].map(disease_cui_id)
        neg_pos_strength_with_id = neg_pos_strength_with_id.rename({'cui_microbe': 'id_microbe', 'cui_disease': 'id_disease'}, axis=1)

        if save_csv:
            microbes_genealogy_with_id.to_csv('{}microbes_genealogy.csv'.format(self.data_folder))
            diseases_genealogy_with_id.to_csv('{}diseases_genealogy.csv'.format(self.data_folder))
            strenghts_with_id.to_csv('{}strengths.csv'.format(self.data_folder))
            neg_pos_strength_with_id.to_csv('{}strengths_neg_pos.csv'.format(self.data_folder))

            microbes_genealogy.to_csv('{}microbes_genealogy_cui.csv'.format(self.data_folder))
            diseases_genealogy.to_csv('{}diseases_genealogy_cui.csv'.format(self.data_folder))
            strenghts.to_csv('{}strengths_cui.csv'.format(self.data_folder))
            neg_pos_strength.to_csv('{}strengths_neg_pos_cui.csv'.format(self.data_folder))


            microbes.to_csv('{}microbes.csv'.format(self.data_folder))
            diseases.to_csv('{}diseases.csv'.format(self.data_folder))

            with open('{}microbes_cui_id.json'.format(self.data_folder), 'w') as f:
                json.dump(microbes_cui_id, f)

            with open('{}diseases_cui_id.json'.format(self.data_folder), 'w') as f:
                json.dump(disease_cui_id, f)



if __name__ == '__main__':
    data_getter = DataGetter()
    data_getter.get_all_data_from_neo4j()

