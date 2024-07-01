import pandas as pd
import numpy as np
import re
import copy

class DataLoader:
    def __init__(self, data_path, use_gold=False, split_gold=True, k=0, seed=0,
                 llm_augmentations_path='', shuffling_augmentations_path='',
                 combine_augmentations_path='', llmRAG_augmentions_path='', use_llm_augmentations=False,
                 use_shuffling_augmentations=False, use_llmRAG_augmentations=False):

        if use_gold:
            self.llm_augmentations = self.preprocess_llm_evidence(pd.read_csv(llm_augmentations_path))
            self.shuffling_augmentations = pd.read_csv(shuffling_augmentations_path)
            self.combined_augmentations = self.preprocess_llm_evidence(pd.read_csv(combine_augmentations_path))

            self.llm_augmentations.index = self.llm_augmentations['Unnamed: 0']
            self.shuffling_augmentations.index = self.shuffling_augmentations['Unnamed: 0']
            self.combined_augmentations.index = self.combined_augmentations['Unnamed: 0']
            del self.llm_augmentations['Unnamed: 0']
            del self.shuffling_augmentations['Unnamed: 0']
            del self.combined_augmentations['Unnamed: 0']

            if 'Unnamed: 0.1' in self.llm_augmentations.columns:
                del self.llm_augmentations['Unnamed: 0.1']
            if 'Unnamed: 0.1' in self.shuffling_augmentations.columns:
                del self.shuffling_augmentations['Unnamed: 0.1']
            if 'Unnamed: 0.1' in self.combined_augmentations.columns:
                del self.combined_augmentations['Unnamed: 0.1']

        self.augmentations = []
        self.rag_augmentations = pd.read_csv(llmRAG_augmentions_path + 'llm_RAG_k{}.csv'.format(k))

        if use_gold and split_gold:
            gold_data = pd.read_csv(data_path)
            np.random.seed(seed)

            # Shuffled data
            gold_data = gold_data.sample(frac=1, axis=0, random_state=seed)
            gold_data_split = self.get_groups(gold_data, n=5)
            #gold_data_split = self.divide_data(gold_data, n=5, seed=seed)
            all_augmentations = self.get_augmentations_split(gold_data_split, use_llm_augmentations=use_llm_augmentations,
                                   use_shuffling_augmentations=use_shuffling_augmentations)
            self.data = gold_data_split[k].astype(str) # Just K
            del self.data['Unnamed: 0']
            self.augmentations = all_augmentations[k].astype(str)

            if use_llm_augmentations:
                self.augmentations = self.preprocess_llm_evidence(pd.read_csv(llm_augmentations_path.replace('.csv', '') + '_k{}.csv'.format(k)))
                #print(self.augmentations)

            self.data_augmented = pd.concat([self.data, self.augmentations], axis=0)

            if use_llmRAG_augmentations:
                self.data_augmented = pd.concat([self.data_augmented, self.rag_augmentations], axis=0)


        elif use_gold:
            self.data = pd.read_csv(data_path).astype(str)
            del self.data['Unnamed: 0']
            self.augmentations = self.get_augmentations(use_llm_augmentations=use_llm_augmentations,
                                                        use_shuffling_augmentations=use_shuffling_augmentations)
            self.data_augmented = pd.concat([self.data, self.augmentations], axis=0)


        else: # Silver data
            silver_data = pd.read_csv(data_path).astype(str)
            silver_data_split = self.divide_data(silver_data, n=5, seed=seed)

            self.data = silver_data_split[k].astype(str)  # Just K


    def preprocess_llm_evidence(self, df):

        new_df = []
        for i in range(len(df)):
            row = df.iloc[i, :].to_frame().transpose()
            evidence = row['EVIDENCE'].values[0]
            matches = re.findall(r'\d+\.\B', evidence)
            if len(matches) > 0:
                evidence = evidence.replace(matches[0], '').strip()
            new_row = copy.deepcopy(row)
            new_row['EVIDENCE'] = evidence
            new_df.append(new_row)

        new_df = pd.concat(new_df, axis=0)

        return new_df

    def get_groups(self, df, n=5):
        len_df = len(df)
        len_chunk = int(len_df / n)
        list_dfs = [df.iloc[i: i + len_chunk] for i in range(0, len_df, len_chunk)]
        if len(list_dfs) > n:
            last_item = list_dfs[-1]
            list_dfs[-2] = pd.concat([list_dfs[-2], last_item], axis=0)
            list_dfs = list_dfs[:-1]
        return list_dfs

    def divide_data(self, df, n=5, seed=0):
        # Divide equitative
        positives = df[df['RELATION'] == 'positive']
        negatives = df[df['RELATION'] == 'negative']
        relates = df[df['RELATION'] == 'relate']
        nas = df[df['RELATION'] == 'na']

        nas_list = self.get_groups(nas, n=n)
        relate_list = self.get_groups(relates, n=n)
        negative_list = self.get_groups(negatives, n=n)

        final_dfs = []
        positive_index = 0
        for i in range(len(nas_list)):
            df1 = pd.concat([nas_list[i], relate_list[i], negative_list[i]], axis=0)
            positives_needed = 220 - df1.shape[0]
            if i == n -1:
                df_pos = positives.iloc[positive_index:]
            else:
                df_pos = positives.iloc[positive_index: positive_index + positives_needed]
            df1 = pd.concat([df_pos, df1], axis=0)
            df1 = df1.sample(frac=1, random_state=seed)
            positive_index += positives_needed
            final_dfs.append(df1)

        return final_dfs

    def get_augmentations_split(self, data_groups, use_llm_augmentations=False, use_shuffling_augmentations=False):
        if use_llm_augmentations == False and use_shuffling_augmentations == False:
            return [pd.DataFrame() for i in range(len(data_groups))]

        # Augmentation indices
        shuffling_index = self.shuffling_augmentations.index
        llm_index = self.llm_augmentations.index
        combined_index = self.combined_augmentations.index

        final_augmentations = []
        for i in range(len(data_groups)):
            main_index = data_groups[i].index

            # Get intersections
            shuffling_intersection = main_index.intersection(shuffling_index)
            llm_intersection = main_index.intersection(llm_index)
            combined_intersection = main_index.intersection(combined_index)

            # Get corresponding dfs
            shuffling_df = self.shuffling_augmentations.loc[shuffling_intersection]
            llm_df = self.llm_augmentations.loc[llm_intersection]
            combined_df = self.combined_augmentations.loc[combined_intersection]

            # Concatenating the different augmentations based on the selectiosn
            if use_llm_augmentations == True and use_shuffling_augmentations == False:
                final_augmentations.append(llm_df)
            elif use_shuffling_augmentations == True and use_llm_augmentations == False:
                final_augmentations.append(shuffling_df)
            else: # Both true
                final_augmentations.append(pd.concat([combined_df], axis=0))

        return final_augmentations

    def get_augmentations(self, use_llm_augmentations=False, use_shuffling_augmentations=False):
        if use_llm_augmentations == False and use_shuffling_augmentations == False:
            return pd.DataFrame()

        elif use_llm_augmentations == True and use_shuffling_augmentations == False:
            return self.llm_augmentations
        elif use_shuffling_augmentations == True and use_llm_augmentations == False:
            return self.shuffling_augmentations
        else:  # Both true
            return pd.concat([self.llm_augmentations, self.shuffling_augmentations, self.combined_augmentations], axis=0)

    def get_disease(self, index=0):
        row = self.data['DISEASE'].iloc[index]
        return row

    def get_microbe(self, index=0):
        row = self.data['MICROBE'].iloc[index]
        return row

    def get_relation(self, index=0):
        row = self.data['RELATION'].iloc[index]
        return row

    def get_evidence(self, index=0):
        row = self.data['EVIDENCE'].iloc[index]
        return row

if __name__ == '__main__':

    data_loader = DataLoader(data_path='../../databases/llms-microbe-disease/data/gold_data_corrected.csv', use_gold=True,
                             split_gold=True, llm_augmentations_path='augmentations/llm_augmentations.csv',
                             shuffling_augmentations_path='augmentations/shuffling_aumentation.csv',
                             llmRAG_augmentions_path='augmentations/zephyr/',
                             combine_augmentations_path='augmentations/shuffling_llm_aumentation.csv',
                             use_llm_augmentations=False, use_shuffling_augmentations=False,
                             use_llmRAG_augmentations=True)
    for i in range(1100):
        index = i
        disease = data_loader.get_disease(index=index)
        microbe = data_loader.get_microbe(index=index)
        evidence = data_loader.get_evidence(index=index)
        relation = data_loader.get_relation(index=index)

        #print('EVIDENCE: {}'.format(evidence))
        #print('DISEASE: {}'.format(disease))
        #print('MICROBE: {}'.format(microbe))
        if relation == 'na':
            print('RELATION: {}: {}'.format(i, relation))