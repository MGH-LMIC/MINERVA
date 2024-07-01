import json
import pandas as pd
import py2neo
from py2neo import Graph, NodeMatcher

class GraphComposer:
    def __init__(self, graph):
        self.graph = graph

    def create_microbe(self, microbe_dict):
        microbe_dict_aux = {}
        for key, value in microbe_dict.items():
            if type(value) == str:
                value = value.replace("'", "").replace('"', '').replace("' ", " ").replace(" '", " ")

            elif type(value) == list:
                value = [elem.replace("'", "").replace('"', '').replace("' ", " ").replace(" '", " ") for elem in value]

            microbe_dict_aux[key] = value
        microbe_dict = microbe_dict_aux

        query = ("CREATE (n:Microbe |-tax_id: '{}', name: '{}', rank: '{}', definition: '{}', official_name: '{}', "
                 "synonyms: '{}', cui: '{}' -|)").format(microbe_dict['tax_id'], microbe_dict['name'].lower(),
                                                         microbe_dict['rank'], microbe_dict['definition'],
                                                         microbe_dict['official_name'], json.dumps(microbe_dict['synonyms']), microbe_dict['cui'])
        query = query.replace('|-', '{').replace('-|', '}')
        self.graph.run(query)

    def create_disease(self, disease_dict):
        disease_dict_aux = {}
        for key, value in disease_dict.items():
            if type(value) == str:
                value = value.replace("'","").replace('"','').replace("' ", " ").replace(" '", " ")

            elif type(value) == list:
                value = [elem.replace("'","").replace('"', '').replace("' ", " ").replace(" '", " ") for elem in value]

            disease_dict_aux[key] = value
        disease_dict = disease_dict_aux

        query = ("CREATE (n:Disease |- name: '{}', snomedct_concept: '{}', cui: '{}', definition: '{}', "
                 "tui: '{}', official_name: '{}', synonyms: '{}' -|)"
                 .format(disease_dict['name'].lower(), disease_dict['snomedct_concept'], disease_dict['cui'],
                         disease_dict['definition'], disease_dict['tui'], disease_dict['official_name'],
                         json.dumps(disease_dict['synonyms'])))

        query = query.replace('|-', '{').replace('-|', '}')
        self.graph.run(query)

    def set_user_properties(self, user_dict, properties):
        set_string = ''
        for property_name, property_value in properties.items():
            set_string += "r.{} = '{}',".format(property_name, property_value)
        set_string = set_string[:-1]  # Delete the last comma

        query = """
                Match (n:User)
                Where n.username = '{}'
                Set {}
                Return n
                """.format(user_dict['name'], set_string)

        self.graph.run(query)
        #print('Property set for {}'.format(user_dict['name']))

    def create_microbe_if_doesnt_exist(self, microbe_dict):
        # Check existence by username
        query = "Match (n: Microbe) Where n.cui='{}' " \
                "return n".format(microbe_dict['cui'])
        result = self.graph.run(query).data()
        if len(result) > 0:
            #print('Microbe {} already exists'.format(microbe_dict['name']))
            return 0
        else:
            self.create_microbe(microbe_dict)
            #print('Microbe {} created'.format(microbe_dict['name']))


    def create_disease_if_doesnt_exist(self, disease_dict):
        # Check existence by username
        query = "Match (n: Disease) Where n.cui='{}' " \
                "return n".format(disease_dict['cui'])
        result = self.graph.run(query).data()
        if len(result) > 0:
            #print('Disease {} already exists'.format(disease_dict['name']))
            return 0
        else:
            self.create_disease(disease_dict)
            #print('Disease {} created'.format(disease_dict['name']))



    def set_property(self, username='111', property='', new_value=''):
        query = "Match (n:User) Where n.username = '{}' " \
                "Set n.{} = '{}' " \
                "Return n.{}".format(username, property, new_value, property)

        self.graph.run(query)
        #print('{} now has {} as {}'.format(username, new_value, property))


    def create_relation(self, rel_dict):
        rel_dict_aux = {}
        for key, value in rel_dict.items():
            if type(value) == str:
                value = value.replace("'", "").replace('"', '').replace("' ", " ").replace(" '", " ")

            elif type(value) == list:
                value = [elem.replace("'", "").replace('"', '').replace("' ", " ").replace(" '", " ") for elem in value]

            rel_dict_aux[key] = value
        rel_dict = rel_dict_aux


        query = """
            Match (microbe:Microbe), (disease:Disease)
            Where microbe.cui='{}' and disease.cui='{}'
            Create (microbe)-[r:{} |- evidence: "{}", journal: "{}", title: "{}", pmid: "{}", pmcid: "{}", publication_year: "{}", impact_factor: "{}", 
             quartile: "{}", issn: "{}", cui_disease: "{}", cui_microbe: "{}", official_name_microbe: "{}" , official_name_disease: "{}", rel_type: "{}"  -|]->(disease)
        """.format(rel_dict['cui_microbe'], rel_dict['cui_disease'], rel_dict['rel_type'].upper(), rel_dict['evidence'], rel_dict['journal'],
                   rel_dict['title'], rel_dict['pmid'], rel_dict['pmcid'], rel_dict['publication_year'], rel_dict['impact_factor'], rel_dict['quartile'],
                   rel_dict['issn'], rel_dict['cui_disease'], rel_dict['cui_microbe'], rel_dict['official_name_microbe'],
                   rel_dict['official_name_disease'], rel_dict['rel_type'])
        query = query.replace('|-', '{').replace('-|', '}')

        self.graph.run(query)

    def create_relation_if_not_exist(self, rel_dict):
        # Check existence by username
        query = "Match (m: Microbe)-[r]-(d:Disease) Where m.cui='{}' and d.cui= '{}' and r.pmid='{}' " \
                "return r".format(rel_dict['cui_microbe'], rel_dict['cui_disease'], rel_dict['pmid'])
        result = self.graph.run(query).data()
        if len(result) > 0:
            #print('Relation {}-{}-{} already exists'.format(rel_dict['cui_microbe'], rel_dict['cui_disease'], rel_dict['pmid']))
            return 0
        else:
            self.create_relation(rel_dict)
            #print('Relation {}-{}-{} created'.format(rel_dict['cui_microbe'], rel_dict['cui_disease'], rel_dict['pmid']))

    def create_microbe_microbe_relation(self, microbe_dict_child, microbe_dict_parent):
        query = """
            Match (microbe_child:Microbe), (microbe_parent:Microbe)
            Where microbe_child.tax_id='{}' and microbe_parent.tax_id='{}'
            Create (microbe_parent)-[r:PARENT]->(microbe_child)
        """.format(microbe_dict_child['tax_id'], microbe_dict_parent['tax_id'])
        query = query.replace('|-', '{').replace('-|', '}')
        if microbe_dict_child['tax_id'] != microbe_dict_parent['tax_id']:
            self.graph.run(query)


    def create_microbe_microbe_relation_if_not_exist(self, microbe_dict_child, microbe_dict_parent):
        # Check existence by username
        query = "Match (m: Microbe)-[r:PARENT]-(n:Microbe) Where m.tax_id='{}' and n.tax_id= '{}' " \
                "return r".format(microbe_dict_child['tax_id'], microbe_dict_parent['tax_id'])
        result = self.graph.run(query).data()
        if len(result) > 0:
            #print('Relation {}-{} already exists'.format(microbe_dict_child['tax_id'], microbe_dict_parent['tax_id']))
            return 0
        else:
            self.create_microbe_microbe_relation(microbe_dict_child, microbe_dict_parent)
            #print('Relation {}-{} created'.format(microbe_dict_child['tax_id'], microbe_dict_parent['tax_id']))



    def create_disease_disease_relation(self, diseases_dict_child, disease_dict_parent):
        query = """
            Match (disease_child:Disease), (disease_parent:Disease)
            Where disease_child.cui='{}' and disease_parent.cui='{}'
            Create (disease_parent)-[r:PARENT]->(disease_child)
        """.format(diseases_dict_child['cui'], disease_dict_parent['cui'])
        query = query.replace('|-', '{').replace('-|', '}')
        if diseases_dict_child['cui'] != disease_dict_parent['cui']:
            self.graph.run(query)


    def create_disease_disease_relation_if_not_exist(self, diseases_dict_child, disease_dict_parent):
        # Check existence by username
        query = "Match (m: Disease)-[r:PARENT]-(n:Disease) Where m.cui='{}' and n.cui= '{}' " \
                "return r".format(diseases_dict_child['cui'], disease_dict_parent['cui'])
        result = self.graph.run(query).data()
        if len(result) > 0:
            #print('Relation {}-{} already exists'.format(diseases_dict_child['name'], disease_dict_parent['name']))
            return 0
        else:
            self.create_disease_disease_relation(diseases_dict_child, disease_dict_parent)
            #print('Relation {}-{} created'.format(diseases_dict_child['name'], disease_dict_parent['name']))


    def get_relations(self, rel_dict):
        # Check existence by username
        #print('----------------------------------------------------------')
        #print(rel_dict)
        query1 = "Match (m: Microbe)-[r:POSITIVE]-(d:Disease) Where m.cui='{}' and d.cui= '{}' " \
                "return r.rel_type as Relation, r.impact_factor as ImpactFactor, r.quartile as Quartile".format(rel_dict['cui_microbe'],
                                                                                                                rel_dict['cui_disease'])
        result1 = self.graph.run(query1).to_data_frame()

        query2 = "Match (m: Microbe)-[r:NEGATIVE]-(d:Disease) Where m.cui='{}' and d.cui= '{}' " \
                 "return r.rel_type as Relation, r.impact_factor as ImpactFactor, r.quartile as Quartile".format(
            rel_dict['cui_microbe'],
            rel_dict['cui_disease'])
        result2 = self.graph.run(query2).to_data_frame()
        #print(result1)
        #print(result2)
        result = pd.concat([result1, result2], axis=0)
        #print(result)
        #print('----------------------------------------------------------')

        return result

    def create_strength_relation(self, rel_dict, strength_dict):
        # Check existence by username
        query = """MATCH (m: Microbe), (d:Disease) Where m.cui='{}' and d.cui= '{}' 
        MERGE (m)-[r:STRENGTH]-(d) 
        ON CREATE
        SET r.rel_type='strength', r.strength_raw= {}, r.strength_IF={}, r.strength_IFQ={}
        ON MATCH
        SET r.rel_type='strength', r.strength_raw= {}, r.strength_IF={}, r.strength_IFQ={}
        return r
        """.format(rel_dict['cui_microbe'], rel_dict['cui_disease'], strength_dict['total_strength_raw'],
        strength_dict['total_strength_IF'], strength_dict['total_strength_IFQ'],
                   strength_dict['total_strength_raw'],
                   strength_dict['total_strength_IF'], strength_dict['total_strength_IFQ'])
        result = self.graph.run(query)

        return result