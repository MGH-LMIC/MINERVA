import pandas as pd
from py2neo import Graph, NodeMatcher
import streamlit as st
from dataclasses import dataclass

@dataclass(frozen=True)
class GraphQueries:
    #def __init__(self):
    graph = Graph(host="0.0.0.0", port="7687", auth=('neo4j', 'your_password'))

    @st.cache_data
    def count_nodes(self, label='Microbe'):
        query = """
        MATCH (m:{})-[:STRENGTH]-(d) 
        RETURN COUNT(DISTINCT m) as count
        """.format(label)
        result = self.graph.run(query).data()

        return result[0]['count']

    @st.cache_data
    def count_papers(self):
        query = """
        MATCH (:Microbe)-[r:NEGATIVE|POSITIVE]->(:Disease)
        RETURN count(DISTINCT r.pmid) AS total_papers
        """
        result = self.graph.run(query).data()
        return result[0]['total_papers']

    @st.cache_data
    def count_relationships(self):
        query = """
                MATCH ()-[r:POSITIVE]->()
                RETURN count(r) AS count
                """
        result = self.graph.run(query).data()
        positives = result[0]['count']

        query = """
                MATCH ()-[r:NEGATIVE]->()
                RETURN count(r) AS count
                """
        result = self.graph.run(query).data()
        negatives = result[0]['count']
        total = positives + negatives
        return total


    @st.cache_data
    def get_microbe_by_property(self, dicto):
        query = """
        Match (m:Microbe) where m.{} = '{}'
        return m.name as name, m.official_name as official_name, m.cui as cui, m.rank as rank, m.tax_id as tax_id, m.definition as definition
        """.format(list(dicto.keys())[0], list(dicto.values())[0])

        result = self.graph.run(query).data()
        return result[0]


    @st.cache_data
    def get_disease_by_property(self, dicto):
        query = """
                Match (d:Disease) where d.{} = '{}'
                return d.name as name, d.official_name as official_name, d.cui as cui, d.tui as tui, d.snomedct_concept as snomedct_concept, d.definition as definition
                """.format(list(dicto.keys())[0], list(dicto.values())[0])

        result = self.graph.run(query).data()
        return result[0]

    @st.cache_data
    def get_relationship_by_microbe_disease(self, m_dicto, d_dicto):
        query = """
        Match (d:Disease)-[r:{}]-(m:Microbe) 
        WHERE d.{} = '{}' AND m.{} = '{}'
        RETURN r.rel_type as Type, r.title as Title, r.pmid as PMID, r.pmcid as PMCID, r.publication_year as Year, r.impact_factor as ImpactFactor, r.evidence as Evidence
        """.format('POSITIVE', list(d_dicto.keys())[0], list(d_dicto.values())[0],
                   list(m_dicto.keys())[0], list(m_dicto.values())[0])

        result_positive = self.graph.run(query).to_data_frame()

        query = """
            Match (d:Disease)-[r:{}]-(m:Microbe) 
            WHERE d.{} = '{}' AND m.{} = '{}'
            RETURN r.rel_type as Type, r.title as Title, r.pmid as PMID, r.pmcid as PMCID, r.publication_year as Year, r.impact_factor as ImpactFactor, r.evidence as Evidence
                """.format('NEGATIVE', list(d_dicto.keys())[0], list(d_dicto.values())[0],
                           list(m_dicto.keys())[0], list(m_dicto.values())[0])

        result_negative = self.graph.run(query).to_data_frame()

        result = pd.concat([result_negative, result_positive], axis=0)
        return result


    @st.cache_data
    def get_strength_by_microbe_disease(self, m_dicto, d_dicto):
        query = """
                Match (d:Disease)-[r:{}]-(m:Microbe) 
                WHERE d.{} = '{}' AND m.{} = '{}'
                RETURN r.strength_raw as Strength, r.strength_IF as Strength_IF, r.strength_IFQ as Strength_IFQ
                """.format('STRENGTH', list(d_dicto.keys())[0], list(d_dicto.values())[0],
                           list(m_dicto.keys())[0], list(m_dicto.values())[0])

        result = self.graph.run(query).to_data_frame()
        return result

    def get_shortest_path_by_microbe_disease(self, m_dicto, d_dicto):
        query = """
                MATCH (m:Microbe),(d:Disease),
                p = shortestPath((m)-[*..15]-(d)) 
                WHERE m.{} = '{}' AND d.{} = '{}'
                RETURN p
                        """.format(list(m_dicto.keys())[0], list(m_dicto.values())[0],
                                   list(d_dicto.keys())[0], list(d_dicto.values())[0])

        result = self.graph.run(query).to_table()
        return result

    @st.cache_data
    def get_microbes_with_more_connections_pos_neg(self, n=10):
        query = """
            MATCH (m:Microbe)-[r:STRENGTH]->(:Disease)
            WITH m, count(r) AS strength_count,
                 count(CASE WHEN r.strength_raw >= 0 THEN 1 END) AS strength_positive,
                 count(CASE WHEN r.strength_raw < 0 THEN 1 END) AS strength_negative
            ORDER BY strength_count DESC
            RETURN m.name AS microbe_name, strength_count, strength_positive, strength_negative
            LIMIT {}
        """.format(n)
        result = self.graph.run(query).to_data_frame()
        return result

    @st.cache_data
    def get_microbes_with_more_references_pos_neg(self, n=10):
        query = """
            MATCH (m:Microbe)-[r:NEGATIVE |POSITIVE]->(:Disease)
            WITH m, count(r) AS total_relations, 
                 count(CASE WHEN type(r) = 'NEGATIVE' THEN 1 END) AS negative_count,
                 count(CASE WHEN type(r) = 'POSITIVE' THEN 1 END) AS positive_count
            ORDER BY total_relations DESC
            RETURN m.name AS microbe_name, total_relations, negative_count, positive_count
            LIMIT {}
        """.format(n)
        result = self.graph.run(query).to_data_frame()
        return result

    @st.cache_data
    def get_diseases_with_more_connections_pos_neg(self, n=10):
        query = """
               MATCH (:Microbe)-[r:STRENGTH]->(m:Disease)
            WITH m, count(r) AS strength_count,
                 count(CASE WHEN r.strength_raw >= 0 THEN 1 END) AS strength_positive,
                 count(CASE WHEN r.strength_raw < 0 THEN 1 END) AS strength_negative
            ORDER BY strength_count DESC
            RETURN m.name AS disease_name, strength_count, strength_positive, strength_negative
            LIMIT {}
                """.format(n)
        result = self.graph.run(query).to_data_frame()
        return result


    @st.cache_data
    def get_diseases_with_more_references_pos_neg(self, n=10):
        query = """
            MATCH (:Microbe)-[r:NEGATIVE |POSITIVE]->(m:Disease)
            WITH m, count(r) AS total_relations, 
                 count(CASE WHEN type(r) = 'NEGATIVE' THEN 1 END) AS negative_count,
                 count(CASE WHEN type(r) = 'POSITIVE' THEN 1 END) AS positive_count
            ORDER BY total_relations DESC
            RETURN m.name AS disease_name, total_relations, negative_count, positive_count
            LIMIT {}
        """.format(n)
        result = self.graph.run(query).to_data_frame()
        return result

    @st.cache_data
    def get_relationships_by_year(self):
        query = """
        MATCH ()-[r:POSITIVE|NEGATIVE]->() 
        RETURN r.publication_year as publication_year, count(r) AS relationship_count
        ORDER BY r.publication_year
        """
        result = self.graph.run(query).to_data_frame()
        return result

    @st.cache_data
    def get_publications_by_year(self):
        query = """
        MATCH ()-[r:POSITIVE|NEGATIVE]->() 
        RETURN r.publication_year as publication_year, count(DISTINCT r.pmid) as publications
        ORDER BY r.publication_year
        """
        result = self.graph.run(query).to_data_frame()
        return result

    @st.cache_data
    def rank_by_positive_strength(self, strength_type='strength_raw', n=10):
        query = """
        MATCH (n1:Microbe)-[r:STRENGTH]-(n2:Disease)
        RETURN n1.name AS Microbe, n2.name AS Disease, r.{} AS Strength
        ORDER BY Strength DESC
        Limit {}
        """.format(strength_type, n)
        result = self.graph.run(query).to_data_frame()
        return result

    @st.cache_data
    def rank_by_negative_strength(self, strength_type='strength_raw', n=10):
        query = """
        MATCH (n1:Microbe)-[r:STRENGTH]-(n2:Disease)
        RETURN n1.name AS Microbe, n2.name AS Disease, r.{} AS Strength
        ORDER BY Strength ASC
        Limit {}
        """.format( strength_type, n)
        result = self.graph.run(query).to_data_frame()
        return result

    @st.cache_data
    def get_more_relevant_papers(self, n=10):
        query = """
        MATCH ()-[r:POSITIVE|NEGATIVE]->()
        WITH r.pmid AS PMID, count(*) AS Frequency, r.title as Title
        ORDER BY Frequency DESC, r.pmid 
        Limit 10
        RETURN Title, PMID, Frequency 
        """
        result = self.graph.run(query).to_data_frame()
        return result

    @st.cache_data
    def get_publications_by_journal(self, n=10):
        query = """
        MATCH (m:Microbe)-[r:NEGATIVE|POSITIVE]->(d:Disease)
        return r.pmid AS PMID, r.journal as Journal
        """
        result = self.graph.run(query).to_data_frame()
        result.index = result['PMID']
        result = result[~result.index.duplicated(keep='first')]
        result = result['Journal'].value_counts().sort_values(ascending=False).iloc[:n].to_frame('counts')
        return result

    @st.cache_data
    def get_all_microbes(self):
        query = """
        MATCH (m:Microbe)-[:STRENGTH]-(:Disease)
        RETURN DISTINCT m.cui AS cui, m.name AS name 
        """
        result = self.graph.run(query).to_data_frame()
        result = result.sort_values('name', ascending=True)
        return result


    @st.cache_data
    def get_all_diseases(self):
        query = """
        MATCH (m:Disease)-[:STRENGTH]-(:Microbe)
        RETURN DISTINCT m.cui AS cui, m.name AS name 
        """
        result = self.graph.run(query).to_data_frame()
        result = result.sort_values('name', ascending=True)
        return result

    def find_one_hop_microbe(self, cui):
        query = """
            MATCH (m:Microbe) // Start with the specified Microbe
            WHERE m.cui = '{}'
            // Define the one-hop traversal pattern
            WITH m 
            MATCH p = (m)-[r:STRENGTH|PARENT*..1]-(n) 
            // UNWIND r as rel 
            WITH p LIMIT 50
            UNWIND relationships(p) AS rel   // <-- Use relationships(p) here
            WITH startNode(rel) AS source, endNode(rel) AS target, rel as edge
            
            // Return the desired properties
            RETURN 
              source.name AS source, 
              labels(source)[0] AS source_type, // Get the first label
              CASE WHEN type(edge) = "PARENT" THEN 'PARENT' ELSE edge.strength_raw END AS relation,
              target.name AS target,
                labels(target)[0] AS target_type // Get the first label
        """.format(cui)
        result = self.graph.run(query).to_data_frame()
        return result

    def find_one_hop_disease(self, cui):
        query = """
            MATCH (m:Disease) // Start with the specified Microbe
            WHERE m.cui = '{}'
            // Define the three-hop traversal pattern
            WITH m 
            MATCH p = (m)-[r:STRENGTH|PARENT*..1]-(n) 
            // UNWIND r as rel 
            WITH p LIMIT 50
            UNWIND relationships(p) AS rel   // <-- Use relationships(p) here
            WITH startNode(rel) AS source, endNode(rel) AS target, rel as edge
            
            // Return the desired properties
            RETURN 
              source.name AS source, 
              labels(source)[0] AS source_type, // Get the first label
              CASE WHEN type(edge) = "PARENT" THEN 'PARENT' ELSE edge.strength_raw END AS relation,
              target.name AS target,
                labels(target)[0] AS target_type // Get the first label
        """.format(cui)
        result = self.graph.run(query).to_data_frame()
        return result


    def get_microbe_relations(self, cui, rel_type='POSIIVE'):
        if rel_type == 'POSITIVE':
            query = """
            MATCH (m:Microbe)-[r:STRENGTH]-(d:Disease)
            WHERE m.cui = '{}' AND r.strength_raw >= 0
            RETURN m.name as microbe_name, d.name as disease_name, r.strength_raw as strength, d.cui as cui
            """.format(cui)
        else:
            query = """
            MATCH (m:Microbe)-[r:STRENGTH]-(d:Disease)
            WHERE m.cui = '{}' AND r.strength_raw < 0
            RETURN m.name as microbe_name, d.name as disease_name, r.strength_raw as strength, d.cui as cui
            """.format(cui)
        result = self.graph.run(query).to_data_frame()
        return result

    def get_disease_relations(self, cui, rel_type='POSIIVE'):
        if rel_type == 'POSITIVE':
            query = """
            MATCH (m:Microbe)-[r:STRENGTH]-(d:Disease)
            WHERE d.cui = '{}' AND r.strength_raw >= 0
            RETURN d.name as disease_name, m.name as microbe_name, r.strength_raw as strength, m.cui as cui
            """.format(cui)
        else:
            query = """
            MATCH (m:Microbe)-[r:STRENGTH]-(d:Disease)
            WHERE d.cui = '{}' AND r.strength_raw < 0
            RETURN d.name as disease_name, m.name as microbe_name, r.strength_raw as strength, m.cui as cui
            """.format(cui)
        result = self.graph.run(query).to_data_frame()
        return result

    def popularity_in_time(self, label='Microbe', cui=''):
        query = """
        MATCH (m:{})-[r:POSITIVE|NEGATIVE]-()
        WHERE m.cui = '{}'
        RETURN r.publication_year as publication_year, count(DISTINCT r.pmid) as publications
        ORDER BY r.publication_year
        """.format(label, cui)
        result = self.graph.run(query).to_data_frame()
        return result

    def get_related_publications_microbe(self, cui=''):
        query = """
            MATCH (m:Microbe)-[r:POSITIVE|NEGATIVE]-()
            WHERE m.cui = '{}'
            RETURN r.pmid as pmid, m.name as microbe, r.official_name_disease as disease, r.rel_type as rel_type,
             r.publication_year as year, r.journal as journal, r.title as title, r.evidence as evidence
            ORDER BY r.publication_year
            """.format(cui)
        result = self.graph.run(query).to_data_frame()
        return result

    def get_related_publications_disease(self, cui=''):
        query = """
            MATCH (m:Disease)-[r:POSITIVE|NEGATIVE]-()
            WHERE m.cui = '{}'
            RETURN r.pmid as pmid, m.name as disease, r.official_name_microbe as microbe, r.rel_type as rel_type,
             r.publication_year as year, r.journal as journal, r.title as title, r.evidence as evidence
            ORDER BY r.publication_year
            """.format(cui)
        result = self.graph.run(query).to_data_frame()
        return result

if __name__ == '__main__':
    querier = GraphQueries()
    print(querier.get_disease_by_property({'name': 'gondii infection'}))
    print(querier.get_relationship_by_microbe_disease(m_dicto={'name': 'Bacteria'}, d_dicto={'name': 'depression'}))
    print(querier.get_strength_by_microbe_disease(m_dicto={'name': 'Bacteria'}, d_dicto={'name': 'depression'}))
    print(querier.get_shortest_path_by_microbe_disease(m_dicto={'cui': 'C1008539'}, d_dicto={'cui': 'C0040558'}))
    print(querier.rank_by_positive_strength())
    print(querier.rank_by_negative_strength())
    print(querier.get_more_relevant_papers())
    print(querier.count_nodes(label='Microbe'))
    print(querier.count_nodes(label='Disease'))
    print(querier.count_papers())
    print(querier.count_relationships())
    print(querier.get_microbes_with_more_connections_pos_neg(n=10))
    print(querier.get_diseases_with_more_connections_pos_neg(n=10))


    print(querier.get_publications_by_journal(n=10))
    print(querier.get_relationships_by_year())
    print(querier.get_publications_by_year())
    print(querier.get_all_microbes())
    print(querier.get_all_diseases())
    print(querier.get_microbe_by_property({'name': 'Bacteria'}))
    print(querier.find_one_hop_microbe(cui='C0038394'))
    print(querier.get_microbe_relations(cui='C0038394', rel_type='POSITIVE'))
    print(querier.get_microbe_relations(cui='C0038394', rel_type='NEGATIVE'))
    print(querier.popularity_in_time(label='Microbe', cui='C0038394'))
    print(querier.get_related_publications(label='Microbe', cui='C0038394'))

