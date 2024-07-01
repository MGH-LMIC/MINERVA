from py2neo import Graph, NodeMatcher
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import numpy as np
import chromadb
import uuid
from chromadb import Documents, EmbeddingFunction, Embeddings
from dataclasses import dataclass
import streamlit as st

class MyEmbeddingFunction(EmbeddingFunction):

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
        self.similarity_model = AutoModel.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext").cuda()

    def __call__(self, input) -> Embeddings:
        # embed the documents somehow
        toks = self.tokenizer.batch_encode_plus(input,
                                           padding="max_length",
                                           max_length=25,
                                           truncation=True,
                                           return_tensors="pt")
        toks_cuda = {}
        for k, v in toks.items():
            toks_cuda[k] = v.cuda()
        embeddings = self.similarity_model(**toks_cuda)[0][:, 0, :].detach().cpu().numpy()  # use CLS representation as the embedding
        return [elem.tolist() for elem in embeddings]

class VectorDBQueries:
    def __init__(self):
        self.graph = Graph(host="0.0.0.0", port="7687", auth=('neo4j', 'your_password'))
        self.chroma_client = chromadb.PersistentClient(path="embeddings/chormadb")
        self.microbes_collection_name = 'microbes_names_embedding'
        self.diseases_collecion_name = 'diseases_names_embedding'

        try:
            self.chroma_client.delete_collection(self.microbes_collection_name)
            self.chroma_client.delete_collection(self.diseases_collecion_name)
            print('Collections deleted')
        except:
            print('Collections existed previously')
            pass

        self.chroma_microbes_collection = self.chroma_client.create_collection(self.microbes_collection_name,
                                                                               metadata={"hnsw:space": "cosine"},
                                                                               embedding_function=MyEmbeddingFunction())
        self.chroma_diseases_collection = self.chroma_client.create_collection(self.diseases_collecion_name,
                                                                               metadata={"hnsw:space": "cosine"},
                                                                               embedding_function=MyEmbeddingFunction())

    def get_microbes_dict(self):
        query = """
                   Match (m:Microbe) Return m.official_name as name, m.cui as cui
                   """
        result = self.graph.run(query).to_data_frame()
        result.index = result['name']
        del result['name']
        result = result.to_dict()['cui']
        return result

    def get_diseases_dict(self):
        query = """
               Match (m:Disease) Return m.official_name as name, m.cui as cui
               """
        result = self.graph.run(query).to_data_frame()
        result.index = result['name']
        del result['name']
        result = result.to_dict()['cui']
        return result

    def get_name_embeddings(self):
        microbes_dict = self.get_microbes_dict()
        microbes_names = list(microbes_dict.keys())
        microbes_cuis = list(microbes_dict.values())

        diseases_dict = self.get_diseases_dict()
        diseases_names = list(diseases_dict.keys())
        diseases_cui = list(diseases_dict.values())
        bs = 32

        # Embed microbes
        print('Embedding Microbes')
        for i in tqdm(np.arange(0, len(microbes_names), bs)):
            names = microbes_names[i:i + bs]
            cuis = microbes_cuis[i:i + bs]

            self.chroma_microbes_collection.add(
                documents=[elem.lower() for elem in names],
                # embeddings=embeddings,
                metadatas=[{'name': names[j], 'cui': cuis[j]} for j in range(len(names))],
                ids=[str(uuid.uuid4()) for j in range(len(names))]
            )

        # Embed diseases
        print('Embedding Diseases')
        for i in tqdm(np.arange(0, len(diseases_names), bs)):
            names = diseases_names[i:i + bs]
            cuis = diseases_cui[i:i + bs]

            self.chroma_diseases_collection.add(
                documents=[elem.lower() for elem in names],
                metadatas=[{'name': names[j], 'cui': cuis[j]} for j in range(len(names))],
                ids=[str(uuid.uuid4()) for j in range(len(names))]
            )


@dataclass(frozen=True)
class MLQueries:
    #def __init__(self, graph_name='myGraph'):
    graph_name='myGraph'
    graph = Graph(host="0.0.0.0", port="7687", auth=('neo4j', 'your_password'))


    @st.cache_data
    def shortest_path(self, node1, node2):

        # Shortest path
        query = ("""
        MATCH (source:{} |-cui: '{}'-|), (target:{} |-cui: '{}'-|)
            CALL gds.shortestPath.dijkstra.stream('{}', |-
                sourceNode: source,
                targetNode: target,
                relationshipTypes: ['STRENGTH_POSITIVE', 'STRENGTH_NEGATIVE', 'PARENT']
            -|)
            YIELD index, sourceNode, targetNode, totalCost, nodeIds, costs, path
            RETURN
                index,
                gds.util.asNode(sourceNode).name AS sourceNodeName,
                gds.util.asNode(targetNode).name AS targetNodeName,
                totalCost,
                [nodeId IN nodeIds | gds.util.asNode(nodeId).name] AS nodeNames,
                [nodeId IN nodeIds | gds.util.asNode(nodeId).cui] AS nodecuis
                //costs,
                //nodes(path) as path
            ORDER BY index
        """.format(node1['label'], node1['cui'], node2['label'], node2['cui'], self.graph_name)
                 .replace('|-', '{').replace('-|', '}'))
        result = self.graph.run(query).to_data_frame()
        cuis = result['nodecuis'].values[0]

        all_results = []
        for i in range(len(cuis) - 1):
            query = """
            Match (n)-[r:PARENT|STRENGTH]-(m)
            Where n.cui='{}' AND m.cui='{}'
            RETURN n.name as Source, labels(n)[0] as SourceType, type(r) as Relation, r.strength_raw as Strength, m.name as Target, labels(m)[0] as TargetType
            """.format(cuis[i], cuis[i+1])
            result = self.graph.run(query).to_data_frame()
            all_results.append(result)

        all_results = pd.concat(all_results)

        # Other paths
        query = """
        MATCH p = (startNode:{} |-cui: '{}'-|)-[:PARENT|STRENGTH*1..5]-(endNode:{} |-cui: '{}'-|)
        WITH p
        LIMIT 5
        UNWIND relationships(p) AS rel
        WITH startNode(rel) AS source, type(rel) AS relation, endNode(rel) AS target, rel as edge
        RETURN source.name AS Source, 
           labels(source)[0] AS SourceType, 
           relation AS Relation, 
           CASE WHEN relation = "PARENT" THEN 1 ELSE edge.strength_raw END AS Strength,
           target.name AS Target, 
           labels(target)[0] AS TargetType
        """.format(node1['label'], node1['cui'], node2['label'], node2['cui']).replace('|-', '{').replace('-|', '}')

        other_paths = self.graph.run(query).to_data_frame()

        all_results = pd.concat([all_results, other_paths], axis=0)

        return all_results


    @st.cache_data
    def find_nearest_neighbors(self, embedding_property='fastRP-embedding', label='Microbe', cui='C0242946'):
        # Find the id of the node
        query = """
        MATCH (m:{})
        WHERE m.cui = '{}'
        RETURN id(m) AS sourceNodeId 
        """.format(label, cui)
        source_node = self.graph.run(query).data()[0]['sourceNodeId']

        # Run the algorithm in stream mode
        query = """
        CALL gds.knn.filtered.stream('{}'
        """.format(self.graph_name)
        query += ', {'
        query_continuation = """
            topK: 10,
            nodeProperties: ['{}'],
            sourceNodeFilter: {},
            targetNodeFilter: '{}',
            sampleRate: 0.5
        """.format(embedding_property, source_node, label)

        query_continuation2 = """
        })
        YIELD node1, node2, similarity
        RETURN  gds.util.asNode(node2).name AS MicrobeTarget, gds.util.asNode(node2).official_name AS MicrobeTargetOfficial, 
        gds.util.asNode(node2).cui AS MicrobeTargetCui, similarity
        ORDER BY similarity DESCENDING
        """
        query = query + query_continuation + query_continuation2
        result = self.graph.run(query).to_data_frame()
        return result

    @st.cache_data
    def get_embeddings(self, embedding_property='fastRP-embedding', label='Microbe'):
        query = """
        MATCH (n:{})
        RETURN n.cui as cui, n.name as name, n.`{}` as embedding
        """.format(label, embedding_property)
        results = self.graph.run(query).to_data_frame()
        return results

    @st.cache_data
    def find_predicted_links_microbe(self, cui='123'):
        query = """
        MATCH (m:Microbe)-[r:STRENGTH_POSITIVE_PRED]-(d:Disease)
        WHERE m.cui='{}'
        Return d.name as disease, r.pred_strength as strength, d.cui as disease_cui
        ORDER BY r.pred_strength DESC 
        LIMIT 10
        """.format(cui)
        positive_results = self.graph.run(query).to_data_frame()


        query = """
        MATCH (m:Microbe)-[r:STRENGTH_NEGATIVE_PRED]-(d:Disease)
        WHERE m.cui='{}'
        RETURN d.name as disease, r.pred_strength as strength, d.cui as disease_cui
        ORDER BY r.pred_strength DESC 
        LIMIT 10
        """.format(cui)
        negative_results = self.graph.run(query).to_data_frame()

        return positive_results, negative_results

    @st.cache_data
    def find_predicted_links_disease(self, cui='123'):
        query = """
        MATCH (m:Microbe)-[r:STRENGTH_POSITIVE_PRED]-(d:Disease)
        WHERE d.cui='{}'
        Return m.name as microbe, r.pred_strength as strength, m.cui as microbe_cui
        ORDER BY r.pred_strength DESC 
        LIMIT 10
        """.format(cui)
        positive_results = self.graph.run(query).to_data_frame()

        query = """
        MATCH (m:Microbe)-[r:STRENGTH_NEGATIVE_PRED]-(d:Disease)
        WHERE d.cui='{}'
        Return m.name as microbe, r.pred_strength as strength, m.cui as micriobe_cui
        ORDER BY r.pred_strength DESC 
        LIMIT 10
        """.format(cui)
        negative_results = self.graph.run(query).to_data_frame()

        return positive_results, negative_results

    @st.cache_data
    def get_clusters(self, kmeans_property='fastRP-kmeans', label='Microbe'):
        query = """
        MATCH (n:{})
        RETURN n.cui as cui, n.name as name, n.`{}` as cluster
        """.format(label, kmeans_property)
        results = self.graph.run(query).to_data_frame()
        return results



if __name__ == '__main__':
    ml = MLQueries()
    vector_db = VectorDBQueries()
    vector_db.get_name_embeddings()

    from sklearn.manifold import TSNE
    import numpy as np
    import pandas as pd
    import pickle
    import umap


    # Writing embeddings to files
    embeddings = ml.get_embeddings(embedding_property='fastRP-embedding', label='Microbe')
    cui = embeddings[['cui', 'name']].values
    tsne = umap.UMAP(n_components=2, random_state=0, n_neighbors=10, metric='euclidean', min_dist=0.1) #TSNE(n_components=2)
    vectors = np.array(embeddings['embedding'].values.tolist())
    print(vectors.shape)
    vectors_tsne = tsne.fit_transform(vectors)
    print('Microbes')
    vectors_tsne = pd.DataFrame(np.concatenate([cui, vectors_tsne], axis=1), columns=['cui', 'name', 'Dim1', 'Dim2'])
    with open('embeddings/fastRP_embeddings_microbe.pkl', 'wb') as f:
        pickle.dump(vectors_tsne, f)

    embeddings = ml.get_embeddings(embedding_property='node2vec-embedding', label='Microbe')
    cui = embeddings[['cui', 'name']].values
    tsne = umap.UMAP(n_components=2, random_state=0, n_neighbors=10, metric='euclidean', min_dist=0.1) #TSNE(n_components=2)
    vectors = np.array(embeddings['embedding'].values.tolist())
    vectors_tsne = tsne.fit_transform(vectors)
    print('Microbes')
    vectors_tsne = pd.DataFrame(np.concatenate([cui, vectors_tsne], axis=1), columns=['cui', 'name', 'Dim1', 'Dim2'])
    with open('embeddings/node2vec_embeddings_microbe.pkl', 'wb') as f:
        pickle.dump(vectors_tsne, f)

    embeddings = ml.get_embeddings(embedding_property='sage-embedding', label='Microbe')
    cui = embeddings[['cui', 'name']].values
    tsne = umap.UMAP(n_components=2, random_state=0, n_neighbors=10, metric='euclidean', min_dist=0.1) #TSNE(n_components=2)
    vectors = np.array(embeddings['embedding'].values.tolist())
    vectors_tsne = tsne.fit_transform(vectors)
    print('Microbes')
    vectors_tsne = pd.DataFrame(np.concatenate([cui, vectors_tsne], axis=1), columns=['cui', 'name', 'Dim1', 'Dim2'])
    with open('embeddings/sage_embeddings_microbe.pkl', 'wb') as f:
        pickle.dump(vectors_tsne, f)

    embeddings = ml.get_embeddings(embedding_property='metapath-embedding', label='Microbe')
    cui = embeddings[['cui', 'name']].values
    tsne = umap.UMAP(n_components=2, random_state=0, n_neighbors=10, metric='euclidean', min_dist=0.1) #TSNE(n_components=2)
    vectors = np.array(embeddings['embedding'].values.tolist())
    vectors_tsne = tsne.fit_transform(vectors)
    print('Microbes')
    vectors_tsne = pd.DataFrame(np.concatenate([cui, vectors_tsne], axis=1), columns=['cui', 'name', 'Dim1', 'Dim2'])
    with open('embeddings/metapath_embeddings_microbe.pkl', 'wb') as f:
        pickle.dump(vectors_tsne, f)



    # Writing embeddings to files
    embeddings = ml.get_embeddings(embedding_property='fastRP-embedding', label='Disease')
    cui = embeddings[['cui', 'name']].values
    tsne = umap.UMAP(n_components=2, random_state=0, n_neighbors=10, metric='euclidean', min_dist=0.1) #TSNE(n_components=2)
    vectors = np.array(embeddings['embedding'].values.tolist())
    vectors_tsne = tsne.fit_transform(vectors)
    print('Diseases')

    vectors_tsne = pd.DataFrame(np.concatenate([cui, vectors_tsne], axis=1), columns=['cui', 'name', 'Dim1', 'Dim2'])
    with open('embeddings/fastRP_embeddings_disease.pkl', 'wb') as f:
        pickle.dump(vectors_tsne, f)

    embeddings = ml.get_embeddings(embedding_property='node2vec-embedding', label='Disease')
    cui = embeddings[['cui', 'name']].values
    tsne = umap.UMAP(n_components=2, random_state=0, n_neighbors=10, metric='euclidean', min_dist=0.1) #TSNE(n_components=2)
    vectors = np.array(embeddings['embedding'].values.tolist())
    vectors_tsne = tsne.fit_transform(vectors)
    print('Diseases')
    vectors_tsne = pd.DataFrame(np.concatenate([cui, vectors_tsne], axis=1), columns=['cui', 'name', 'Dim1', 'Dim2'])
    with open('embeddings/node2vec_embeddings_disease.pkl', 'wb') as f:
        pickle.dump(vectors_tsne, f)

    embeddings = ml.get_embeddings(embedding_property='sage-embedding', label='Disease')
    cui = embeddings[['cui', 'name']].values
    tsne = umap.UMAP(n_components=2, random_state=0, n_neighbors=10, metric='euclidean', min_dist=0.1) #TSNE(n_components=2)
    vectors = np.array(embeddings['embedding'].values.tolist())
    vectors_tsne = tsne.fit_transform(vectors)
    print('Diseases')
    vectors_tsne = pd.DataFrame(np.concatenate([cui, vectors_tsne], axis=1), columns=['cui', 'name', 'Dim1', 'Dim2'])
    with open('embeddings/sage_embeddings_disease.pkl', 'wb') as f:
        pickle.dump(vectors_tsne, f)


    embeddings = ml.get_embeddings(embedding_property='metapath-embedding', label='Disease')
    cui = embeddings[['cui', 'name']].values
    tsne = umap.UMAP(n_components=2, random_state=0, n_neighbors=10, metric='euclidean', min_dist=0.1) #TSNE(n_components=2)
    vectors = np.array(embeddings['embedding'].values.tolist())
    vectors_tsne = tsne.fit_transform(vectors)
    print('Diseases')
    vectors_tsne = pd.DataFrame(np.concatenate([cui, vectors_tsne], axis=1), columns=['cui', 'name', 'Dim1', 'Dim2'])
    with open('embeddings/metapath_embeddings_disease.pkl', 'wb') as f:
        pickle.dump(vectors_tsne, f)

