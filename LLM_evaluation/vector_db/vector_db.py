import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import pandas as pd
from langchain.vectorstores.utils import maximal_marginal_relevance
import numpy as np
import  torch


class MyVectorDB:
    def __init__(self, db_dir='vector_db_files/', chroma_collection='silver_data',
                 embedding_func='default', kw_file='', model_name='', masked=False):
        self.chroma_client = chromadb.PersistentClient(db_dir)
        embedding_model = self.get_embedding_func(embedding_func)
        self.db_name = chroma_collection
        self.db_name_kw = '{}_kw_'.format(model_name) + chroma_collection
        self.kw_file = kw_file

        self.chroma_collection = self.chroma_client.get_or_create_collection(name=self.db_name,
                                                                             embedding_function=embedding_model,
                                                                             metadata={"hnsw:space": "cosine"})

        self.chroma_collection_kw = self.chroma_client.get_or_create_collection(name=self.db_name_kw,
                                                                             embedding_function=embedding_model,
                                                                             metadata={"hnsw:space": "cosine"})

        # Re-Ranker
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        self.masked = masked

    def get_embedding_func(self, embedding_func):
        if embedding_func == 'default':
            emb_func = embedding_functions.DefaultEmbeddingFunction()
        else:
            emb_func = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=embedding_func)

        return emb_func

    def divide_data(self, df, n=5):
        len_df = len(df)
        len_chunk = int(len_df/n)

        list_dfs = [df.iloc[i: i + len_chunk] for i in range(0, len_df, len_chunk)]
        return list_dfs

    def transform_kw(self, kw_dict):
        kw_dict_aux = {}
        for k, v in kw_dict.items():
            new_k = (k[0], k[1], k[2])
            kw_dict_aux[new_k] = v
        return kw_dict_aux

    def save_files(self, data_base_path='databases/llms-microbe-disease/data/data_MDI_SilverDataforTrain.xlsx', use_gold=False, seed=0, k=0):
        if use_gold:
            gold_data = pd.read_csv(data_base_path)
            np.random.seed(seed)

            # Shuffled data
            gold_data = gold_data.sample(frac=1, axis=0, random_state=seed)
            gold_data_split = self.divide_data(gold_data, n=5)
            self.data = pd.concat([gold_data_split[i] for i in range(len(gold_data_split)) if i != k], axis=0) # Everything but k

        else:
            self.data = pd.read_excel(data_base_path)


        diseases_list = []
        microbes_list = []
        relations_list = []
        evidences_list = []
        keywords_list = []
        keywords_indices = []
        for i in range(len(self.data)):
            row = self.data.iloc[i, :]
            diseases_list.append(row['DISEASE'])
            microbes_list.append(row['MICROBE'])
            relations_list.append(row['RELATION'])
            evidences_list.append(row['EVIDENCE'])


        print('Total N° of rows: {}'.format(len(self.data)))
        print('Total N° of Diseases: {}'.format(len(list(set(diseases_list)))))
        print('Total N° of Microbes: {}'.format(len(list(set(microbes_list)))))

        metadata_list = [{'disease': str(diseases_list[i]), 'microbe': str(microbes_list[i]), 'evidence': evidences_list[i],
                          'relation': str(relations_list[i])} for i in range(len(evidences_list))]
        ids_list = ['{}'.format(i) for i in range(len(evidences_list))]


        print('Saving everything to main chroma db')
        self.chroma_collection.add(
            documents=evidences_list,
            metadatas=metadata_list,
            ids=ids_list)




    def only_one_mode(self, collection, sentence, n_neighbors=3, use_mmr=True, fetch_k=50, lambda_mult=0.5,
                      use_reranker=False):
        if use_mmr:
            embedding_sentence = collection._embedding_function([sentence])
            similarity_results = collection.query(
                query_embeddings=embedding_sentence,
                n_results=fetch_k,
                include=["metadatas", "documents", "distances", "embeddings"])

            result_embeddings = similarity_results['embeddings'][0]
            metadatas = similarity_results['metadatas'][0]

            mmr_results = maximal_marginal_relevance(
                np.array(embedding_sentence, dtype=np.float32),
                result_embeddings,
                k=n_neighbors,
                lambda_mult=lambda_mult)

            results = [r for i, r in enumerate(metadatas) if i in mmr_results]


        elif use_reranker:
            results = collection.query(
                query_texts=[sentence],
                include=['documents', 'metadatas'],
                n_results=fetch_k)

            documents = results['documents'][0]
            metadatas = results['metadatas'][0]

            # Using Reranker
            cross_inp = [[sentence, documents[i]] for i in range(len(documents))]
            cross_scores = self.cross_encoder.predict(cross_inp)
            cross_scores = {i: cross_scores[i] for i in range(len(cross_scores))}
            cross_scores = sorted(cross_scores.items(), key=lambda x: x[1], reverse=True)
            cross_scores_indices = [elem[0] for elem in cross_scores][:n_neighbors] # Just the first set neighbors

            results = [metadatas[cross_scores_indices[i]] for i in range(len(cross_scores_indices))]

        else:
            results =collection.query(
                query_texts=[sentence],
                include=['documents', 'metadatas'],
                n_results=n_neighbors)

            results = results['metadatas'][0]

        return results


    def get_neighbors(self, sentence='airways are clear', disease='', microbe='', keywords='', modality='', n_neighbors=3, use_mmr=True, fetch_k=50, lambda_mult=0.5,
                      use_reranker=False):
        """
         lambda_mult: Number between 0 and 1 that determines the degree
                        of diversity among the results with 0 corresponding
                        to maximum diversity and 1 to minimum diversity.
                        Defaults to 0.5.
        """
        collections_dict = {'sentence': self.chroma_collection, 'keywords': self.chroma_collection_kw}

        if modality == 'sentence':
            results = self.only_one_mode(collections_dict[modality], sentence, n_neighbors=n_neighbors, use_mmr=use_mmr,
                                         fetch_k=fetch_k, lambda_mult=lambda_mult, use_reranker=use_reranker)
        elif modality == 'keywords':
            results = self.only_one_mode(collections_dict[modality], keywords, n_neighbors=n_neighbors, use_mmr=use_mmr,
                                         fetch_k=fetch_k, lambda_mult=lambda_mult, use_reranker=use_reranker)

        else: # modality == combined
            n_neighbors_each = int(n_neighbors/2)
            results_sentence = self.only_one_mode(collections_dict['sentence'], sentence, n_neighbors=n_neighbors_each, use_mmr=use_mmr,
                                         fetch_k=fetch_k, lambda_mult=lambda_mult, use_reranker=use_reranker)

            results_keywords = self.only_one_mode(collections_dict['keywords'], keywords, n_neighbors=n_neighbors_each, use_mmr=use_mmr,
                                         fetch_k=fetch_k, lambda_mult=lambda_mult, use_reranker=use_reranker)
            results = []
            for i in range(len(results_sentence)):
                results.append(results_keywords[i])
                results.append(results_sentence[i])

        return results


if __name__ == '__main__':

    # Gold data vector db
    model_name = 'biomistral'
    for k in range(5):
        db_name = 'gold_data_k{}'.format(k)

        print('Creating vector db for gold data {}'.format(k + 1))

        my_vector_db = MyVectorDB(chroma_collection=db_name, model_name=model_name)  # silver_data, silver_data_masked
        my_vector_db.save_files(data_base_path='../initial_db/gold_data_corrected.csv',
                                use_gold=True, seed=0, k=k)


    my_vector_db = MyVectorDB(chroma_collection='gold_data_k0', model_name=model_name)  # silver_data, silver_data_masked


    results = my_vector_db.get_neighbors(sentence='the microbe increases the probability of diabetes',
                                         keywords='increases', modality='sentence', disease='diabetes',
                                         microbe='microbe', n_neighbors=4, use_mmr=False, fetch_k=50, lambda_mult=0.5,
                                         use_reranker=True)
    print('SENTENCE RESULTS:')
    for elem in results:
        print(elem)
    print('-'*50)
    print('')



