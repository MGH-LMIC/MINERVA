from py2neo import Graph, NodeMatcher
import pandas as pd
from collections import defaultdict

class GraphComparator:
    def __init__(self, db_path):
        self.db_path = db_path
        self.db_file = pd.read_csv(db_path)
        self.folder = db_path.split('/')[0] + '/'
        self.db_name = db_path.split('/')[0]

        # Graph initialization
        self.graph = Graph(host="0.0.0.0", port="7687", auth=('neo4j', 'synhodo123'))

    def default_list(self):
        return []

    def default_num(self):
        return 0

    def query_relation(self, d_cui, b_cui, bacteria_name, disease_name):
        query = """Match (d:Disease)-[r:STRENGTH]-(b:Microbe) 
                    where d.cui='{}' and b.cui='{}' 
                    Return d.official_name as Disease, b.official_name as Bacteria, r.strength_raw as StrengthRaw, 
                    r.strength_IF as StrengthIF""".format(d_cui, b_cui)
        result = self.graph.run(query).data()

        # Try with all combinations
        if len(result) == 0:
            query_parents_microbe = """
            MATCH (m:Microbe)-[:PARENT*2]-(relatedMicrobes:Microbe)
                Where m.cui='{}'
                RETURN DISTINCT relatedMicrobes.cui as cui
            """.format(b_cui)
            result_microbes = self.graph.run(query_parents_microbe).data()

            query_parents_disease = """
                        MATCH (d:Disease)-[:PARENT*2]-(relatedDiseases:Disease)
                            Where d.cui='{}'
                            RETURN DISTINCT relatedDiseases.cui as cui
                        """.format(d_cui)
            result_diseases = self.graph.run(query_parents_disease).data()

            for m in result_microbes:
                for d in result_diseases:
                    query = """Match (d:Disease)-[r:STRENGTH]-(b:Microbe) 
                                       where d.cui='{}' and b.cui='{}' 
                                       Return d.official_name as Disease, b.official_name as Bacteria, r.strength_raw as StrengthRaw, 
                                       r.strength_IF as StrengthIF""".format(d['cui'], m['cui'])
                    result = self.graph.run(query).data()
                    if len(result) > 0:
                        d_cui = d['cui']
                        b_cui = m['cui']
                        bacteria_name = result[0]['Bacteria']
                        disease_name = result[0]['Disease']
                        break
                if len(result) > 0:
                    break

        if len(result) > 0:
            if result[0]['StrengthRaw'] > 0:
                query = """Match (d:Disease)-[r:POSITIVE]-(b:Microbe) 
                                    where d.cui='{}' and b.cui='{}' 
                                    Return d.official_name as Disease, b.official_name as Bacteria, r.evidence as Evidence, r.pmid as PMID 
                                    """.format(d_cui, b_cui)

                result_pos = self.graph.run(query).to_data_frame()
                if len(result_pos) > 0:
                    pmids = result_pos['PMID'].values
                    result_pos = result_pos['Evidence'].values
                    evidence = ' ||| \n'.join(
                        ['[{}] {}'.format(pmids[i], result_pos[i]) for i in range(min(len(result_pos), 10))])
                else:
                    evidence = ''
                relations_of_class = len(result_pos)
            else:

                query = """Match (b:Microbe)-[r:NEGATIVE]-(d:Disease) 
                                                    where d.cui='{}' and b.cui='{}' 
                                                    Return d.official_name as Disease, b.official_name as Bacteria, r.evidence as Evidence, r.pmid as PMID 
                                                    """.format(d_cui, b_cui)
                result_neg = self.graph.run(query).to_data_frame()
                if len(result_neg) > 0:
                    pmids = result_neg['PMID'].values
                    result_neg = result_neg['Evidence'].values
                    evidence = ' ||| \n'.join(
                        ['[{}] {}'.format(pmids[i], result_neg[i]) for i in range(min(len(result_neg), 10))])
                else:
                    evidence = ''

                relations_of_class = len(result_neg)
        else:
            evidence = ''
            relations_of_class = 0

        return result, evidence, relations_of_class, d_cui, b_cui, bacteria_name, disease_name


    def query_bacteria_disease(self, d_cui, b_cui):
        query_bacteria = """Match (n:Microbe) WHERE n.cui='{}' Return n.name""".format(b_cui)
        result_bacteria = self.graph.run(query_bacteria).data()

        query_disease = """Match (n:Disease) WHERE n.cui='{}' Return n.name""".format(d_cui)
        result_disease = self.graph.run(query_disease).data()

        bacteria_present = False
        disease_present = False
        if len(result_bacteria) > 0:
            bacteria_present = True

        if len(result_disease) > 0:
            disease_present = True

        return bacteria_present, disease_present


    def compare(self):
        total_relations = 0
        not_considered_relations = 0
        repeated_relations = 0
        existing_relations = 0
        coinciding_relations = 0
        coinciding_relations_IF = 0
        unsure_relations = 0
        unsure_relations_IF = 0
        wrong_relations = defaultdict(self.default_list)
        correct_relations = defaultdict(self.default_list)
        examined_relations = {}
        bacterias_dict = {}
        disease_dict = {}

        for i in range(len(self.db_file)):
            row = self.db_file.iloc[i]

            if i % 200 == 0 and i != 0:
                print('{}: {}|{}'.format(self.db_name, i, len(self.db_file)))
                print('Summary of results for {}'.format(self.db_path))
                print('Total relations: {}'.format(total_relations))
                print('Not considered relations: {}'.format(not_considered_relations))
                print('Repeated relations: {}'.format(repeated_relations))
                print('Coinciding bacterias: {} ({} %)'.format(len(bacterias_dict), sum(list(bacterias_dict.values()))*100/len(bacterias_dict)))
                print('Coinciding diseases: {} ({} %)'.format(len(disease_dict), sum(list(disease_dict.values()))*100/len(disease_dict)))
                print('Existing relations: {} ({} %)'.format(existing_relations, round(existing_relations*100/total_relations, 2)))
                print('Coinciding relations: {} ({} %)'.format(coinciding_relations, round(coinciding_relations*100/existing_relations, 2)))
                print('Unsure relations: {} ({} %)'.format(unsure_relations, round(unsure_relations*100/existing_relations, 2)))
                print('Coinciding relations IF: {} ({} %)'.format(coinciding_relations_IF, round(coinciding_relations_IF*100/existing_relations, 2)))
                print('Unsure relations IF: {} ({} %)'.format(unsure_relations_IF, round(unsure_relations_IF*100/existing_relations, 2)))
                print('--------------------------------------------')

            bacteria = str(row['b_cui'])
            disease = str(row['d_cui'])
            bacteria_name = str(row['b_name'])
            disease_name = str(row['d_name'])
            relation = str(row['relation'])
            evidence = str(row['evidence'])
            original_relation = str(row['original_relation'])
            paper = str(row['paper'])

            if bacteria == 'nan' or disease == 'nan':
                not_considered_relations += 1
                continue

            # Eliminate repeated relations
            if (bacteria, disease) in examined_relations.keys():
                repeated_relations += 1
                continue
            else:
                examined_relations[(bacteria, disease)] = 1

            bacteria_present, disease_present = self.query_bacteria_disease(disease, bacteria)
            bacterias_dict[bacteria] = int(bacteria_present)
            disease_dict[disease] = int(disease_present)

            # Just count positive and negative relations
            if relation == 'positive' or relation == 'negative':
                total_relations += 1
            #########################################################################################
            #else:
            #    continue
            #########################################################################################

            if int(bacteria_present) > 0 and int(disease_present) > 0:
                neo_relation, neo_evidence, relations_of_class, disease_out, bacteria_out, bacteria_name_out, disease_name_out = (
                    self.query_relation(b_cui=bacteria, d_cui=disease, bacteria_name=bacteria_name, disease_name=disease_name))
            else:
                neo_relation = []
                neo_evidence = ''
                relations_of_class = 0
                bacteria_out = bacteria
                disease_out = disease
                bacteria_name_out = bacteria_name
                disease_name_out = disease_name

            # Check if relation exists
            if len(neo_relation) > 0:
                existing_relations += 1

                # If relation exists Check if coincide
                neo_direction = neo_relation[0]['StrengthRaw']
                neo_direction_IF = neo_relation[0]['StrengthIF']

                if neo_direction == 0:
                    unsure_relations += 1

                elif neo_direction > 0 and relation == 'positive':
                    coinciding_relations += 1
                    correct_relations['neo_relation'].append('positive')
                    correct_relations['neo_strength'].append(neo_direction)
                    correct_relations['relations_of_class'].append(relations_of_class)
                    correct_relations['db_relation'].append(relation)
                    correct_relations['db_original_relation'].append(original_relation)
                    correct_relations['neo_evidence'].append(neo_evidence)
                    correct_relations['db_evidence'].append(evidence)
                    correct_relations['paper'].append(paper)
                    correct_relations['b_cui'].append(bacteria)
                    correct_relations['b_name'].append(bacteria_name)
                    correct_relations['d_cui'].append(disease)
                    correct_relations['d_name'].append(disease_name)
                    if bacteria != bacteria_out:
                        correct_relations['b_cui_out'].append(bacteria_out)
                        correct_relations['b_name_out'].append(bacteria_name_out)
                    else:
                        correct_relations['b_cui_out'].append(bacteria)
                        correct_relations['b_name_out'].append('')
                    if disease != disease_out:
                        correct_relations['d_cui_out'].append(disease_out)
                        correct_relations['d_name_out'].append(disease_name_out)
                    else:
                        correct_relations['d_cui_out'].append(disease)
                        correct_relations['d_name_out'].append('')


                elif neo_direction < 0 and relation == 'negative':
                    coinciding_relations += 1
                    correct_relations['neo_relation'].append('negative')
                    correct_relations['neo_strength'].append(neo_direction)
                    correct_relations['relations_of_class'].append(relations_of_class)
                    correct_relations['db_relation'].append(relation)
                    correct_relations['db_original_relation'].append(original_relation)
                    correct_relations['neo_evidence'].append(neo_evidence)
                    correct_relations['db_evidence'].append(evidence)
                    correct_relations['paper'].append(paper)
                    correct_relations['b_cui'].append(bacteria)
                    correct_relations['b_name'].append(bacteria_name)
                    correct_relations['d_cui'].append(disease)
                    correct_relations['d_name'].append(disease_name)
                    if bacteria != bacteria_out:
                        correct_relations['b_cui_out'].append(bacteria_out)
                        correct_relations['b_name_out'].append(bacteria_name_out)
                    else:
                        correct_relations['b_cui_out'].append(bacteria)
                        correct_relations['b_name_out'].append('')
                    if disease != disease_out:
                        correct_relations['d_cui_out'].append(disease_out)
                        correct_relations['d_name_out'].append(disease_name_out)
                    else:
                        correct_relations['d_cui_out'].append(disease)
                        correct_relations['d_name_out'].append('')

                else:
                    if neo_direction > 0:
                        neo_relation_aux = 'positive'
                    else:
                        neo_relation_aux = 'negative'
                    wrong_relations['neo_relation'].append(neo_relation_aux)
                    wrong_relations['neo_strength'].append(neo_direction)
                    wrong_relations['relations_of_class'].append(relations_of_class)
                    wrong_relations['db_relation'].append(relation)
                    wrong_relations['db_original_relation'].append(original_relation)
                    wrong_relations['neo_evidence'].append(neo_evidence)
                    wrong_relations['db_evidence'].append(evidence)
                    wrong_relations['paper'].append(paper)
                    wrong_relations['b_cui'].append(bacteria)
                    wrong_relations['b_name'].append(bacteria_name)
                    wrong_relations['d_cui'].append(disease)
                    wrong_relations['d_name'].append(disease_name)
                    if bacteria != bacteria_out:
                        wrong_relations['b_cui_out'].append(bacteria_out)
                        wrong_relations['b_name_out'].append(bacteria_name_out)
                    else:
                        wrong_relations['b_cui_out'].append(bacteria)
                        wrong_relations['b_name_out'].append('')
                    if disease != disease_out:
                        wrong_relations['d_cui_out'].append(disease_out)
                        wrong_relations['d_name_out'].append(disease_name_out)
                    else:
                        wrong_relations['d_cui_out'].append(disease)
                        wrong_relations['d_name_out'].append('')

                if neo_direction_IF > 0 and relation == 'positive':
                    coinciding_relations_IF += 1
                elif neo_direction_IF < 0 and relation == 'negative':
                    coinciding_relations_IF += 1
                elif neo_direction_IF == 0:
                    unsure_relations_IF += 1
                else:  # Wrong
                    pass


        wrong_relations = pd.DataFrame(wrong_relations)
        wrong_relations.to_csv(self.folder + '{}_wrong.csv'.format(self.db_name))

        correct_relations = pd.DataFrame(correct_relations)
        correct_relations.to_csv(self.folder + '{}_correct.csv'.format(self.db_name))


        # Summarization of the results

        print('Summary of results for {}'.format(self.db_path))
        print('Total relations: {}'.format(total_relations))
        print('Not considered relations: {}'.format(not_considered_relations))
        print('Repeated relations: {}'.format(repeated_relations))
        print('Coinciding bacterias: {} ({} %)'.format(len(bacterias_dict), sum(list(bacterias_dict.values()))*100/len(bacterias_dict)))
        print('Coinciding diseases: {} ({} %)'.format(len(disease_dict), sum(list(disease_dict.values()))*100/len(disease_dict)))
        print('Existing relations: {} ({} %)'.format(existing_relations, round(existing_relations*100/total_relations, 2)))
        print('Coinciding relations: {} ({} %)'.format(coinciding_relations, round(coinciding_relations*100/existing_relations, 2)))
        print('Unsure relations: {} ({} %)'.format(unsure_relations, round(unsure_relations*100/existing_relations, 2)))
        print('Coinciding relations IF: {} ({} %)'.format(coinciding_relations_IF, round(coinciding_relations_IF*100/existing_relations, 2)))
        print('Unsure relations IF: {} ({} %)'.format(unsure_relations_IF, round(unsure_relations_IF*100/existing_relations, 2)))
        print('')


if __name__ == '__main__':
    db_paths = ['AMADIS/AMADIS_newNoRepeated.csv', 'GMDAD/GMDAD_newNoRepeated.csv', 'HMDAD/HMDAD_newNoRepeated.csv',
                'Original/Original_newNoRepeated.csv',
                'Disbiome/Disbiome_newNoRepeated.csv']

    for db_path in db_paths:
        graph_comparator = GraphComparator(db_path=db_path)
        graph_comparator.compare()