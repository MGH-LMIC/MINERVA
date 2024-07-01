import pandas as pd
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.encode_relations = {'positive': 0, 'negative': 1, 'relate':2, 'na':3}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx, :]
        disease = row['DISEASE']
        microbe = row['MICROBE']
        relation = row['RELATION']
        evidence = row['EVIDENCE']
        question = row['QUESTIONS']

        input_to_network = evidence + ' [SEP] ' + question
        relation = self.encode_relations[relation]
        sample = {'inputs': input_to_network, 'relations': relation}
        return sample

if __name__ == '__main__':
    from new_release_paper.normal_db.data_loader import DataLoader as MyDataloader

    data_path = '../LLM_evaluation/initial_db/gold_data_corrected.csv'
    use_gold = True
    split_gold = True
    data_loader1 = MyDataloader(data_path=data_path, use_gold=use_gold, split_gold=split_gold, k=0).data
    data_loader2 = MyDataloader(data_path=data_path, use_gold=use_gold, split_gold=split_gold, k=1).data
    data_loader3 = MyDataloader(data_path=data_path, use_gold=use_gold, split_gold=split_gold, k=2).data
    data_loader4 = MyDataloader(data_path=data_path, use_gold=use_gold, split_gold=split_gold, k=3).data
    data_loader5 = MyDataloader(data_path=data_path, use_gold=use_gold, split_gold=split_gold, k=4).data

    datasets = {0: {'train': pd.concat([data_loader2, data_loader3, data_loader4, data_loader5], axis=0), 'validation': data_loader1},
                1: {'train': pd.concat([data_loader1, data_loader3, data_loader4, data_loader5], axis=0), 'validation': data_loader2},
                2: {'train': pd.concat([data_loader1, data_loader2, data_loader4, data_loader5], axis=0), 'validation': data_loader3},
                3: {'train': pd.concat([data_loader1, data_loader2, data_loader3, data_loader5], axis=0), 'validation': data_loader4},
                4: {'train': pd.concat([data_loader1, data_loader2, data_loader3, data_loader4], axis=0), 'validation': data_loader5}}

    for k in range(5):

        train_dataset = datasets[k]['train']
        val_dataset = datasets[k]['validation']
        #print(train_dataset)

        train_dataset = MyDataset(train_dataset)
        val_dataset = MyDataset(val_dataset)
        train_dataloader = DataLoader(train_dataset, batch_size=16,
                            shuffle=True, num_workers=0)
        val_dataloader = DataLoader(val_dataset, batch_size=16,
                            shuffle=True, num_workers=0)
        for i_batch, sample_batched in enumerate(train_dataloader):
            print(sample_batched)

