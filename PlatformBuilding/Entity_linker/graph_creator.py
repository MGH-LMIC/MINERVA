import numpy as np
import pandas as pd
from py2neo import Graph, NodeMatcher
from metapub import PubMedFetcher
from impact_factor.core import Factor
import taxopy
from grap_composer import GraphComposer
from scispacy.linking import EntityLinker
from owlready2 import get_ontology, default_world
from transformers import AutoTokenizer, AutoModel
from scipy.spatial.distance import cosine as CosineDistance
from medspacy.ner import TargetRule, TargetMatcher
import spacy

default_world.set_backend(filename="../utils/pym.sqlite3")

class GraphCreator:
    def __init__(self, score_strategy='none', default_impact=3.5, default_quartile=2):
        # Make the connection to Neo4jDB
        self.graph = Graph(host="0.0.0.0", port="7687", auth=('neo4j', 'your_pasword'))
        self.graph_composer = GraphComposer(self.graph)

        # Impact factor
        self.factor = Factor()

        # Taxdb
        self.taxdb = taxopy.TaxDb(nodes_dmp="../utils/taxdump/nodes.dmp", names_dmp="../utils/taxdump/names.dmp")

        # Diseases
        self.diseases_db = get_ontology("http://PYM/").load()
        self.SNOMEDCT_CODES = self.diseases_db["SNOMEDCT_US"]
        self.CUI_CODES = self.diseases_db["CUI"]

        self.diseases_similarity_tokenizer = AutoTokenizer.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
        self.diseases_similarity_model = AutoModel.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext").cuda()
        self.diseases_linker = EntityLinker(linker_name='umls', k=1)

        # Bacterias
        self.bacteria_finder = spacy.load("en_core_sci_sm")
        self.bacteria_finder.remove_pipe('ner')
        self.bacteria_linker = EntityLinker(nlp=self.bacteria_finder, resolve_abbreviations=True, linker_name='umls')

        # Score calculation
        self.score_strategy = score_strategy
        self.default_impact = default_impact
        self.default_quartile = default_quartile

    def get_parents_diseases(self, concept):
        parents = concept.parents

        parents_line = []
        for initial_parent in parents:
            cui = initial_parent >> self.CUI_CODES
            cui = list(cui)[0].name
            try:
                cui_info = self.diseases_linker.kb.cui_to_entity[cui]
                definition = cui_info[4]
                tui = cui_info[3][0]
                official_name = cui_info[1]
                synonyms = cui_info[2][:5]
            except KeyError:
                definition = None
                tui = None
                official_name = None
                synonyms = None
            parent_line = [{'snomedct_concept': initial_parent.name, 'cui': cui, 'definition': definition,
                            'tui': tui, 'official_name': official_name, 'synonyms':synonyms, 'name': str(initial_parent.label[0]).lower()}]
            continue_line = True
            while continue_line:
                try:
                    parent = initial_parent.parents[0]
                    cui = parent >> self.CUI_CODES
                    cui = list(cui)[0].name
                    try:
                        cui_info = self.diseases_linker.kb.cui_to_entity[cui]
                        definition = cui_info[4]
                        tui = cui_info[3][0]
                        official_name = cui_info[1]
                        synonyms = cui_info[2][:5]
                    except KeyError:
                        definition = None
                        tui = None
                        official_name = None
                        synonyms = None
                    parent_line.append({'snomedct_concept': parent.name, 'cui': cui, 'definition': definition,
                                        'tui': tui, 'official_name': official_name, 'synonyms':synonyms, 'name': str(parent.label[0]).lower()})
                    initial_parent = parent
                except IndexError:
                    continue_line = False
            parents_line.append(parent_line)

        return parents_line

    def get_children_diseases(self, concept):
        children = concept.children

        children_line = []
        for initial_child in children:
            cui = initial_child >> self.CUI_CODES
            cui = list(cui)[0].name
            try:
                cui_info = self.diseases_linker.kb.cui_to_entity[cui]
                definition = cui_info[4]
                tui = cui_info[3][0]
                official_name = cui_info[1]
                synonyms = cui_info[2][:5]
            except KeyError:
                definition = None
                tui = None
                official_name = None
                synonyms = None
            child_line = [{'snomedct_concept': initial_child.name, 'cui': cui, 'definition': definition,
                           'tui': tui, 'official_name': official_name, 'synonyms':synonyms, 'name': str(initial_child.label[0]).lower()}]
            continue_line = True
            while continue_line:
                try:
                    child = initial_child.children[0]
                    cui = child >> self.CUI_CODES
                    cui = list(cui)[0].name
                    try:
                        cui_info = self.diseases_linker.kb.cui_to_entity[cui]
                        definition = cui_info[4]
                        tui = cui_info[3][0]
                        official_name = cui_info[1]
                        synonyms = cui_info[2][:5]
                    except KeyError:
                        definition = None
                        tui = None
                        official_name = None
                        synonyms = None
                    child_line.append({'snomedct_concept': child.name, 'cui': cui, 'definition': definition,
                                       'tui': tui, 'official_name': official_name, 'synonyms':synonyms, 'name': str(child.label[0]).lower()})
                    initial_child = child
                except IndexError:
                    continue_line = False
            children_line.append(child_line)

        return children_line


    def get_family_diseases(self, diseases_info):
        diseas_cui = diseases_info['cui']
        try:
            diseases_snomedct = self.CUI_CODES[diseas_cui] >> self.SNOMEDCT_CODES
        except TypeError:
            closest_concept = None
            parents = []
            children = []
            return closest_concept, parents, children

        diseases_snomedct = list(diseases_snomedct)

        # If there are more than one similar concept
        if len(diseases_snomedct) > 1:
            all_embs = []
            labels = [str(elem.label[0]) for elem in diseases_snomedct]
            labels = [diseases_info['official_name']] + labels

            toks = self.diseases_similarity_tokenizer.batch_encode_plus(labels,
                                               padding="max_length",
                                               max_length=30,
                                               truncation=True,
                                               return_tensors="pt")
            toks_cuda = {}
            for k, v in toks.items():
                toks_cuda[k] = v.cuda()
            cls_rep = self.diseases_similarity_model(**toks_cuda)[0][:, 0, :]
            all_embs.append(cls_rep.cpu().detach().numpy())
            all_embs = np.concatenate(all_embs, axis=0)

            target = all_embs[0, :]
            other_vectors = all_embs[1:, :]
            min_distance = 100000
            index = -1
            for i in range(other_vectors.shape[0]):
                dist = CosineDistance(target, other_vectors[i])
                if dist < min_distance:
                    min_distance = dist
                    index = i
            closest_concept = diseases_snomedct[index]

        elif len(diseases_snomedct) == 1:
            closest_concept = diseases_snomedct[0]

        else: # Equal to 0
            closest_concept = None

        if closest_concept != None:
            parents = self.get_parents_diseases(closest_concept)
            children = self.get_children_diseases(closest_concept)
            closest_concept = closest_concept.name
        else:
            parents = []
            children = []

        return closest_concept, parents, children

    def create_disease_nodes(self, main_disease, parents, children):
        # Main disease first
        self.graph_composer.create_disease_if_doesnt_exist(main_disease)

        # Relation with all the parents
        for parent_line in parents:
            root_disease = main_disease
            for i in range(len(parent_line)):
                parent = parent_line[i]
                self.graph_composer.create_disease_if_doesnt_exist(parent)
                self.graph_composer.create_disease_disease_relation_if_not_exist(diseases_dict_child=root_disease,
                                                                                 disease_dict_parent=parent)
                root_disease = parent

        # Relation with all childs
        for child_line in children:
            root_disease = main_disease
            for i in range(len(child_line)):
                child = child_line[i]
                self.graph_composer.create_disease_if_doesnt_exist(child)
                self.graph_composer.create_disease_disease_relation_if_not_exist(diseases_dict_child=child,
                                                                                 disease_dict_parent=root_disease)
                root_disease = child


    def get_family_bacterias(self, bacteria_info):

        # Get Bacteria ID
        taxid = taxopy.taxid_from_name(bacteria_info['official_name'], self.taxdb)
        if taxid == []:
            for synonym in bacteria_info['synonyms']:
                taxid = taxopy.taxid_from_name(synonym, self.taxdb)
                if taxid == []:
                    taxid = taxopy.taxid_from_name(synonym.lower(), self.taxdb)

                if taxid != []:
                    break
            if taxid == []:
                taxid = None
                bacteria_dict = {'tax_id': None, 'name': bacteria_info['name'].lower(), 'rank': None,
                                 'definition': bacteria_info['definition'],
                                 'official_name': bacteria_info['official_name'],
                                 'synonyms': bacteria_info['synonyms'], 'cui': bacteria_info['cui']}
                return bacteria_dict, []

            else:
                taxid = taxid[0]
                info = taxopy.Taxon(taxid, self.taxdb)
                bacteria_lineage = info.ranked_name_lineage
                bacteria_dict = {'tax_id': info.taxid, 'name': bacteria_info['name'].lower(), 'rank': info.rank,
                                 'definition': bacteria_info['definition'],
                                 'official_name': bacteria_info['official_name'],
                                 'synonyms': bacteria_info['synonyms'], 'cui': bacteria_info['cui']}
        else:
            taxid = taxid[0]
            info = taxopy.Taxon(taxid, self.taxdb)
            bacteria_lineage = info.ranked_name_lineage

            bacteria_dict = {'tax_id': info.taxid, 'name': bacteria_info['name'].lower(), 'rank': info.rank,
                         'definition': bacteria_info['definition'], 'official_name': bacteria_info['official_name'],
                         'synonyms': bacteria_info['synonyms'], 'cui': bacteria_info['cui']}

        # Catching the lineage
        lineage_dicts = []
        self.bacteria_ruler = TargetMatcher(self.bacteria_finder)
        new_rules = []
        for elem in bacteria_lineage:
            taxid = taxopy.taxid_from_name(elem[1], self.taxdb)
            rank = elem[0]
            name = elem[1].lower()

            if len(taxid) > 0:
                new_rules += [TargetRule(literal=name, category='BACTERIA')]
                lineage_dicts.append({'tax_id': taxid[0], 'rank': rank, 'name': name})
        self.bacteria_ruler.add(new_rules)

        lineage_dicts_new = []
        for dicto in lineage_dicts:
            doc = self.bacteria_finder(dicto['name'])
            doc = self.bacteria_ruler(doc)
            doc = self.bacteria_linker(doc)
            try:
                entity = doc.ents[0]
                umls_ent = entity._.kb_ents[0]
                cui = umls_ent[0]
                info = self.bacteria_linker.kb.cui_to_entity[cui]
                tui = info[3][0]
                definition = info[4]
                official_name = info[1]
                synonyms = info[2][:5]
            except IndexError:
                cui = None
                tui = None
                definition = None
                official_name = None
                synonyms = None

            dicto['cui'] = cui
            dicto['tui'] = tui
            dicto['definition'] = definition
            dicto['official_name'] = official_name
            dicto['synonyms'] = synonyms

            lineage_dicts_new.append(dicto)

        return bacteria_dict, lineage_dicts_new


    def create_bacteria_nodes(self, bacteria_dict, bacteria_family):
        # Node of Main Bacteria
        self.graph_composer.create_microbe_if_doesnt_exist(bacteria_dict)

        # Create microbe lineages
        for elem in bacteria_family:
            self.graph_composer.create_microbe_if_doesnt_exist(elem)

        for i in range(len(bacteria_family) - 1):
            self.graph_composer.create_microbe_microbe_relation_if_not_exist(bacteria_family[i], bacteria_family[i + 1])


    def create_relationships(self, relation_dict):
        # First, create the relationship if doesn't exist
        self.graph_composer.create_relation_if_not_exist(relation_dict)


    def create_strength_relations(self, rel_dict):
        relations = self.graph_composer.get_relations(rel_dict)
        if relations.shape[0] == 0:
            return 0
        #print(relations)
        # Calculating the strength of the relation

        # Raw strength
        rel_types = relations['Relation'].values
        rel_types = [1 if elem == 'positive' else -1 for elem in rel_types]
        total_strength_raw = sum(rel_types)

        # Using impact factor
        rel_types = relations['Relation'].values
        rel_types = [1 if elem == 'positive' else -1 for elem in rel_types]
        impact_factors = [np.log10(float(elem)) if elem != 'NA' else self.default_impact for elem in relations['ImpactFactor'].values]
        total_strength_if = round(sum([rel_types[i]*impact_factors[i] for i in range(len(impact_factors))]), 3)

        # Usinf IF and queartilesd
        rel_types = relations['Relation'].values
        rel_types = [1 if elem == 'positive' else -1 for elem in rel_types]
        impact_factors = [elem if elem != 'NA' else self.default_impact for elem in relations['ImpactFactor'].values]
        quartiles = [int(elem[1]) if elem !='NA' else self.default_quartile for elem in relations['Quartile'].values]
        total_strength_ifq = round(sum([rel_types[i]*np.log10(float(impact_factors[i])/quartiles[i]) for i in range(len(impact_factors))]), 3)

        # Create strength relation
        strength_dict = {'total_strength_raw': total_strength_raw, 'total_strength_IF': total_strength_if,
                         'total_strength_IFQ': total_strength_ifq}
        self.graph_composer.create_strength_relation(rel_dict, strength_dict)


    def create_relation(self, paper_info, disease_info, bacteria_info, evidence, relation):
        # We skip processing if we find these kind of relations
        if relation == 'na' or relation == 'relate':
            return


        # Disease processing
        closest_concept, parents, children = self.get_family_diseases(disease_info)
        disease_info['snomedct_concept'] = closest_concept
        self.create_disease_nodes(disease_info, parents, children)

        # Taxon processing
        bacteria_dict, bacteria_family = self.get_family_bacterias(bacteria_info)
        self.create_bacteria_nodes(bacteria_dict, bacteria_family)


        # Relationship Processing
        issn = paper_info['issn']

        try:
            factor = self.factor.search(issn)
            i_factor = factor[0]['factor']
            if i_factor == '':
                i_factor = 'NA'
            quartile = factor[0]['jcr']
            if quartile == '':
                quartile = 'NA'
        except:
            i_factor = 'NA'
            quartile = 'NA'

        relation_dict = {'cui_disease': disease_info['cui'], 'cui_microbe': bacteria_dict['cui'],
                         'tax_id_microbe': bacteria_dict['tax_id'], 'snomedct_concept_disease': disease_info['snomedct_concept'],
                         'rel_type': relation, 'evidence': evidence, 'journal':paper_info['journal'],
                         'title': paper_info['title'], 'pmid': paper_info['pubmed_id'], 'pmcid': paper_info['pmc_id'],
                         'publication_year': paper_info['publication_date'], 'official_name_disease': disease_info['official_name'],
                         'official_name_microbe': bacteria_info['official_name'],
                         'impact_factor': i_factor, 'quartile': quartile, 'issn': issn}
        self.create_relationships(relation_dict)


        # Strength relations
        self.create_strength_relations(relation_dict)




if __name__ == '__main__':
    create_graph = GraphCreator()
    #create_graph.check_intersections()
    create_graph.create_relation()