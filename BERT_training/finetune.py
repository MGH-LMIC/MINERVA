import numpy as np
import torch
from new_release_paper.normal_db.data_loader import DataLoader as MyDataloader
from create_dataset import MyDataset
from transformers import AutoTokenizer, AutoModel
from model import Model
import pandas as pd
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from transformers import AdamW
from transformers.optimization import get_linear_schedule_with_warmup
from pytorch_metric_learning.losses import NTXentLoss

device = torch.device('cuda')


class FinetuneBert:
    def __init__(self, model, lr=1e-5, weight_decay=0.001, model_name='', encode_relations={}, folder='',
                 weights=None, alpha_contrastive=0):
        self.model = model
        self.model_name = model_name
        self.folder = folder
        self.encode_relations = encode_relations
        self.optimizer = AdamW(self.model.parameters(), lr=lr)#optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=10, num_training_steps=50)
        self.criterion = nn.CrossEntropyLoss(weight=weights)
        self.constrastive_criterion = NTXentLoss(temperature=0.7)
        self.alpha_contrastive = alpha_contrastive

        self.min_val = 1000
        self.no_better = 0
        self.patience = 5

    def train(self, train_dataloader, valid_dataloader, epochs=100, k=0):
        for epoch in range(epochs):
            self.model.train()
            train_losses = []
            for i_batch, sample_batched in enumerate(tqdm(train_dataloader)):
                inputs = sample_batched['inputs']
                relations = sample_batched['relations'].to(device)

                # Optimization loop
                self.optimizer.zero_grad()
                predictions, embeddings = self.model(inputs)
                loss = self.criterion(predictions, relations)
                if self.alpha_contrastive > 0:
                    loss_contrastive = self.constrastive_criterion(embeddings, relations)
                    loss = loss + self.alpha_contrastive*loss_contrastive

                train_losses.append(loss.item())
                loss.backward()
                self.optimizer.step()
            self.scheduler.step()
            train_losses = np.array(train_losses).mean()

            with torch.no_grad():
                self.model.eval()
                val_losses = []
                for i_batch, sample_batched in enumerate(tqdm(valid_dataloader)):
                    inputs = sample_batched['inputs']
                    relations = sample_batched['relations'].to(device)

                    # Optimization loop
                    predictions, embeddings = self.model(inputs)
                    loss = self.criterion(predictions, relations)
                    val_losses.append(loss.item())

                val_losses = np.array(val_losses).mean()
                print('k:{} | Epoch: {} | Train Loss: {} | Val Loss: {}'.format(k, epoch, train_losses, val_losses))

                if val_losses < self.min_val:
                    self.min_val = val_losses
                    self.no_better = 0
                    print('Best Model')
                    torch.save(self.model, '{}/{}_best.pt'.format(self.folder, self.model_name))
                    #self.model.model.save_pretrained('{}/{}_backbone_best.pt'.format(self.folder, self.model_name))
                else:
                    self.no_better += 1
                    if self.no_better >= self.patience:
                        print('Finished by Early Stopping')
                        return


    def test(self, test_dataloader, k=0):
        self.model = torch.load('{}/{}_best.pt'.format(self.folder, self.model_name))
        with torch.no_grad():
            self.model.eval()
            test_losses = []
            all_preds = []
            all_targets = []
            for i_batch, sample_batched in enumerate(test_dataloader):
                inputs = sample_batched['inputs']
                relations = sample_batched['relations'].to(device)

                # Optimization loop
                predictions, embeddings = self.model(inputs)
                predicted_class = F.softmax(predictions, dim=1)
                predicted_class = torch.argmax(predicted_class, dim=1)

                all_targets.append(relations.cpu().numpy().squeeze())
                all_preds.append(predicted_class.cpu().numpy().squeeze())
                loss = self.criterion(predictions, relations)
                test_losses.append(loss.item())

            test_losses = np.array(test_losses).mean()
            print('Test Loss: {}'.format(test_losses))

            all_preds = np.concatenate(all_preds)
            all_targets = np.concatenate(all_targets)

            report = classification_report(y_true=all_targets, y_pred=all_preds,
                                           target_names=self.encode_relations.keys(), output_dict=True)
            print('Report for k:{}'.format(k))
            print(report)
            torch.save(report, '{}/{}_report.pkl'.format(self.folder, self.model_name))

if __name__ == '__main__':
    use_gold = True
    split_gold = True
    batch_size = 32
    epochs = 100
    model_to_finetune = 'michiyasunaga/BioLinkBERT-base'
    model_to_finetune = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
    model_name = model_to_finetune.split('/')[1]
    device = torch.device('cuda')

    data_path = '../databases/llms-microbe-disease/data/gold_data_corrected.csv'
    data_loader1 = MyDataloader(data_path=data_path, use_gold=use_gold, split_gold=split_gold, k=0).data
    data_loader2 = MyDataloader(data_path=data_path, use_gold=use_gold, split_gold=split_gold, k=1).data
    data_loader3 = MyDataloader(data_path=data_path, use_gold=use_gold, split_gold=split_gold, k=2).data
    data_loader4 = MyDataloader(data_path=data_path, use_gold=use_gold, split_gold=split_gold, k=3).data
    data_loader5 = MyDataloader(data_path=data_path, use_gold=use_gold, split_gold=split_gold, k=4).data

    datasets = {0: {'train': pd.concat([data_loader3, data_loader4, data_loader5], axis=0),
                    'test': data_loader1, 'validation': data_loader2},
                1: {'train': pd.concat([data_loader1, data_loader4, data_loader5], axis=0),
                    'test': data_loader2, 'validation': data_loader3},
                2: {'train': pd.concat([data_loader1, data_loader2, data_loader5], axis=0),
                    'test': data_loader3, 'validation': data_loader4},
                3: {'train': pd.concat([data_loader1, data_loader2, data_loader3], axis=0),
                    'test': data_loader4, 'validation': data_loader5},
                4: {'train': pd.concat([data_loader2, data_loader3, data_loader4], axis=0),
                    'test': data_loader5, 'validation': data_loader1}}


    for k in range(5):

        train_dataset = MyDataset(datasets[k]['train'])
        val_dataset = MyDataset(datasets[k]['validation'])
        test_dataset = MyDataset(datasets[k]['test'])

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                      shuffle=True, num_workers=0)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size,
                                    shuffle=True, num_workers=0)

        test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                                    shuffle=True, num_workers=0)

        model = AutoModel.from_pretrained(model_to_finetune)
        tokenizer = AutoTokenizer.from_pretrained(model_to_finetune)
        my_model = Model(model, tokenizer, device=device).to(device)

        finetuner = FinetuneBert(model=my_model, model_name=model_name + '_k{}_simp'.format(k), encode_relations=train_dataset.encode_relations)

        finetuner.train(train_dataloader, val_dataloader, epochs=epochs, k=k)
        finetuner.test(test_dataloader, k=k)
