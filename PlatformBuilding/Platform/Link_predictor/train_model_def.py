import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from torch_geometric.data import HeteroData
from torch_geometric.transforms import Compose, ToUndirected, RandomLinkSplit, RemoveIsolatedNodes, RemoveDuplicatedEdges
from my_model import Model
from torch_geometric.loader import LinkNeighborLoader, NeighborLoader
from tqdm import tqdm
import torch.nn as nn
import json
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from codify_other_dbs import OtherDBsCodifier
import copy
import time
from comet_ml import Experiment
from dataset_creator import DatasetCreator
import os
import torch.nn.functional as F

class Trainer:
    def __init__(self, dataset_creator, folder='results/', patience=50, experiment_name='GNN',
                 hyperparameters={}):

        self.dataset_creator = dataset_creator
        self.seed = self.dataset_creator.seed
        self.patience = patience
        self.min_val = 10000
        self.no_better = 0

        self.experiment = Experiment(api_key="SN8b7ORCZ8BjlES0bvWGBslAC",
                                     project_name="microbiome",
                                     workspace="salangarica")

        self.experiment_name = experiment_name
        self.folder = folder
        self.experiment.set_name(self.experiment_name)

        if hyperparameters != {}:
            self.experiment.add_tag('neighs_{}_hops_{}'.format(hyperparameters['num_neighbors'],
                                                               len(hyperparameters['num_neighbors'])))

            self.experiment.add_tag('Net_size{}'.format(hyperparameters['network_size']))
            self.experiment.add_tag('{}'.format(hyperparameters['task']))

            if hyperparameters['use_negative_examples']:
                self.experiment.add_tag('Neg')
            if hyperparameters['binary_classification']:
                self.experiment.add_tag('Bin')


    def split_dataset(self, dataset, edge_type, ratio=[0.7, 0.15, 0.15], use_otherDBs_mask=True, masks=[]):
        if use_otherDBs_mask and masks != []:
            edge_index = dataset[edge_type].edge_index
            train_mask, val_mask, test_mask = masks
            train_mask = torch.from_numpy(train_mask)
            val_mask = torch.from_numpy(val_mask)
            test_mask = torch.from_numpy(test_mask)

            # Create split edge indices:
            train_edge_index = edge_index[:, train_mask]
            val_edge_index = edge_index[:, val_mask]
            test_edge_index = edge_index[:, test_mask]

            # Construct HeteroData objects for each split:
            train_data = dataset.edge_subgraph({edge_type: train_mask})
            train_data[edge_type].edge_index = train_edge_index

            val_data = dataset.edge_subgraph({edge_type: val_mask})
            val_data[edge_type].edge_index = val_edge_index

            test_data = dataset.edge_subgraph({edge_type: test_mask})
            test_data[edge_type].edge_index = test_edge_index

            return train_data, val_data, test_data

        else:
            edge_index = dataset[edge_type].edge_index

            num_edges = edge_index.size(1)
            perm = torch.randperm(num_edges)
            num_val = int(ratio[1] * num_edges)
            num_test = int(ratio[2] * num_edges)

            train_mask = perm[num_val + num_test:]
            val_mask = perm[:num_val]
            test_mask = perm[num_val:num_val + num_test]

            # Create split edge indices:
            train_edge_index = edge_index[:, train_mask]
            val_edge_index = edge_index[:, val_mask]
            test_edge_index = edge_index[:, test_mask]


            # Construct HeteroData objects for each split:
            train_data = dataset.edge_subgraph({edge_type: train_mask})
            train_data[edge_type].edge_index = train_edge_index

            val_data = dataset.edge_subgraph({edge_type: val_mask})
            val_data[edge_type].edge_index = val_edge_index

            test_data = dataset.edge_subgraph({edge_type: test_mask})
            test_data[edge_type].edge_index = test_edge_index

            return train_data, val_data, test_data

    def dataset_preprocessing(self, dataset, add_homogenous_edges=False,
                              num_neighbors=[5, 5, 5, 5], batch_size=32):
        dataset = ToUndirected()(dataset)
        del dataset['diseases', 'rev_strengths', 'microbes'].edge_label  # Remove "reverse" label.

        if add_homogenous_edges:
            dicto_neighs = {('microbes', 'strengths', 'diseases'): num_neighbors,
                            ('diseases', 'rev_strengths', 'microbes'): num_neighbors,
                            ('microbes', 'parent_m', 'microbes'): num_neighbors,
                            ('diseases', 'parent_d', 'diseases'): num_neighbors}
        else:
            dicto_neighs = {('microbes', 'strengths', 'diseases'): num_neighbors,
                           ('diseases', 'rev_strengths', 'microbes'): num_neighbors}

        ### Dataset Operations ###
        kwargs = dict(  # Shared data loader arguments:
            data=dataset,
            num_neighbors=dicto_neighs,
            batch_size=batch_size,
            num_workers=4,
            persistent_workers=True,
        )
        edge_label_index = dataset['microbes', 'diseases'].edge_index
        edge_label = dataset['microbes', 'diseases'].edge_label

        dataloader = LinkNeighborLoader(
                dataset=dataset,
                edge_label_index=(('microbes', 'diseases'), edge_label_index),
                edge_label=edge_label,
                # neg_sampling=dict(mode='binary', amount=5),
                shuffle=True,
                **kwargs,
            )

        return dataset, dataloader


    def train_regressor(self, data, strength_th=0, add_node_features=False, use_otherDBs_mask=True,
                        add_homogenous_edges=True, network_size=32, num_neighbors=[5, 5, 5, 5], batch_size=32,
                        inner_product_dec=False, learning_rate=0.01,
                        add_homogenous_edges_embedding=True, negatives_ratio=0.5,
                        embedding='no', embedding_epochs=10, **kwargs):

        # Create the hetero_data
        if inner_product_dec:
            training_type_inter = 'classification'
        else:
            training_type_inter = 'regression'

        dataset, masks = self.dataset_creator.create_dataset(data, strength_th=strength_th,
                                      training_type=training_type_inter, use_negatives_examples=False,
                                                             add_homogenous_edges=add_homogenous_edges,
                                                             add_homogenous_edges_embedding=add_homogenous_edges_embedding,
                                                             embedding=embedding, embedding_epochs=embedding_epochs)

        # Split dataset
        train_dataset, val_dataset, test_dataset = self.split_dataset(dataset, edge_type=('microbes', 'strengths', 'diseases'),
                                                                      use_otherDBs_mask=use_otherDBs_mask, masks=masks)

        # Preprocessing
        train_dataset, train_dataloader = self.dataset_preprocessing(train_dataset
                                                                     ,add_homogenous_edges=add_homogenous_edges,
                                                                     num_neighbors=num_neighbors, batch_size=batch_size)
        val_dataset, val_dataloader = self.dataset_preprocessing(val_dataset
                                                                 ,add_homogenous_edges=add_homogenous_edges,
                                                                     num_neighbors=num_neighbors, batch_size=batch_size)

        test_dataset, test_dataloader = self.dataset_preprocessing(test_dataset
                                                                   ,add_homogenous_edges=add_homogenous_edges,
                                                                     num_neighbors=num_neighbors, batch_size=batch_size)


        # Defining the model
        model = Model(hidden_channels=network_size, data=train_dataset,
                      inner_product_dec=inner_product_dec).to(device).float()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        if inner_product_dec:
            criterion = nn.BCEWithLogitsLoss()
        else:
            criterion = nn.MSELoss()

        # Training
        for epoch in range(300):
            train_loss = []
            train_y_binary = []
            train_out_binary = []

            val_loss = []
            val_y_binary = []
            val_out_binary = []

            test_loss = []
            test_y_binary = []
            test_out_binary = []

            model.train()
            for batch_idx, batch in enumerate(tqdm(train_dataloader)):
                optimizer.zero_grad()
                batch = batch.to(device)

                out = model(
                    batch.x_dict,
                    batch.edge_index_dict,
                    batch['microbes', 'diseases'].edge_label_index,
                ).squeeze()
                y = batch['microbes', 'diseases'].edge_label.float()

                loss = criterion(out, y.float())

                loss.backward()
                optimizer.step()
                torch.cuda.empty_cache()

                # Classification_labels
                if inner_product_dec:
                    out = F.sigmoid(out)
                    train_out_binary.append(out.round(decimals=0).detach().cpu().numpy().squeeze())
                else:
                    train_out_binary.append(torch.sign(out).detach().cpu().numpy().squeeze())
                train_y_binary.append(y.cpu().numpy().squeeze())
                train_loss.append(loss.item())

            train_loss = np.array(train_loss).mean()

            train_y_binary = np.concatenate(train_y_binary)
            train_out_binary = np.concatenate(train_out_binary)
            train_accuracy = accuracy_score(train_y_binary, train_out_binary)
            train_f1 = f1_score(train_y_binary, train_out_binary)
            train_precision = precision_score(train_y_binary, train_out_binary)
            train_recall = recall_score(train_y_binary, train_out_binary)

            print('Epoch {}: Train Loss: {} | Acc: {} | f1: {} | Precision: {} | Recall: {}'.format(epoch, train_loss,
                                                                                                    train_accuracy, train_f1,
                                                                                                    train_precision, train_recall))

            self.experiment.log_metrics({'train_loss': train_loss, 'train_accuracy': train_accuracy,
                                         'train_f1': train_f1, 'train_precision': train_precision, 'train_recall': train_recall}, epoch=epoch)


            # Validation
            with torch.no_grad():
                model.eval()
                for batch_idx, batch in enumerate(tqdm(val_dataloader)):
                    batch = batch.to(device)

                    out = model(
                        batch.x_dict,
                        batch.edge_index_dict,
                        batch['microbes', 'diseases'].edge_label_index,
                    ).squeeze()
                    y = batch['microbes', 'diseases'].edge_label.float()

                    loss = criterion(out, y.float())

                    # Classification_labels
                    if inner_product_dec:
                        out = F.sigmoid(out)
                        val_out_binary.append(out.round(decimals=0).detach().cpu().numpy().squeeze())
                    else:
                        val_out_binary.append(torch.sign(out).detach().cpu().numpy().squeeze())
                    val_y_binary.append(y.cpu().numpy().squeeze())
                    val_loss.append(loss.item())

            val_loss = np.array(val_loss).mean()

            val_y_binary = np.concatenate(val_y_binary)
            val_out_binary = np.concatenate(val_out_binary)

            val_accuracy = accuracy_score(val_y_binary, val_out_binary)
            val_f1 = f1_score(val_y_binary, val_out_binary)
            val_precision = precision_score(val_y_binary, val_out_binary)
            val_recall = recall_score(val_y_binary, val_out_binary)

            print('Epoch {}: val Loss: {} | Acc: {} | f1: {} | Precision: {} | Recall: {}'.format(epoch, val_loss,
                                                                                                    val_accuracy,
                                                                                                    val_f1,
                                                                                                    val_precision,
                                                                                                    val_recall))

            self.experiment.log_metrics({'val_loss': val_loss, 'val_accuracy': val_accuracy,
                                         'val_f1': val_f1, 'val_precision': val_precision,
                                         'val_recall': val_recall}, epoch=epoch)

            if val_loss < self.min_val:
                self.min_val = val_loss
                self.no_better = 0
                torch.save(model.state_dict(), os.path.join(self.folder, 'best_model.pt'))
            else:
                self.no_better += 1
                if self.no_better > self.patience:
                    print('Stopping by Early Stopping')
                    return 0

            # Testing
            test_loss = []
            test_out_binary = []
            test_y_binary = []
            test_probabilities = []
            with torch.no_grad():
                test_dataset = test_dataset.to(device)
                out = model(
                    test_dataset.x_dict,
                    test_dataset.edge_index_dict,
                    test_dataset['microbes', 'diseases'].edge_index,
                )

                y = test_dataset['microbes', 'diseases'].edge_label

                loss = criterion(out, y.float())

                if inner_product_dec:
                    out = F.sigmoid(out)
                    test_out_binary.append(out.round(decimals=0).detach().cpu().numpy().squeeze())
                else:
                    test_out_binary.append(torch.sign(out).detach().cpu().numpy().squeeze())
                test_loss.append(loss.item())
                test_probabilities.append(out.detach().cpu().numpy().squeeze())
                test_y_binary.append(y.cpu().numpy().squeeze())

            test_loss = np.array(test_loss).mean()
            test_y_binary = np.concatenate(test_y_binary)
            test_out_binary = np.concatenate(test_out_binary)
            test_probabilities = np.concatenate(test_probabilities)

            test_accuracy = accuracy_score(test_y_binary, test_out_binary)
            test_f1 = f1_score(test_y_binary, test_out_binary, average='weighted')
            test_precision = precision_score(test_y_binary, test_out_binary, average='weighted')
            test_recall = recall_score(test_y_binary, test_out_binary, average='weighted')

            print('Epoch {}: test Loss: {} | Acc: {} | f1: {} | Precision: {} | Recall: {}'.format(epoch,
                                                                                                   test_loss,
                                                                                                   test_accuracy,
                                                                                                   test_f1,
                                                                                                   test_precision,
                                                                                                   test_recall))

            self.experiment.log_metrics({'test_loss': test_loss, 'test_accuracy': test_accuracy,
                                         'test_f1': test_f1, 'test_precision': test_precision,
                                         'test_recall': test_recall}, epoch=epoch)

            # Indices
            edge_indices = test_dataset['microbes', 'diseases'].edge_index.cpu().numpy()
            edge_microbes = edge_indices[0, :]
            edge_diseases = edge_indices[1, :]

            all_preds = []
            for i in range(len(edge_microbes)):
                e_microbe = edge_microbes[i]
                e_disease = edge_diseases[i]
                gt = test_y_binary[i]
                pred = test_out_binary[i]
                probabilities = test_probabilities[i]

                e_microbe_df = data['microbes'][data['microbes']['id'] == e_microbe][
                    ['id', 'cui', 'name']].reset_index(drop=True)
                e_microbe_df.columns = ['id_microbe', 'cui_microbe', 'name_microbe']

                e_disease_df = data['diseases'][data['diseases']['id'] == e_disease][
                    ['id', 'cui', 'name']].reset_index(drop=True)
                e_disease_df.columns = ['id_disease', 'cui_disease', 'name_disease']

                # Appended
                df = pd.concat([e_microbe_df, e_disease_df], axis=1)
                df['GT'] = gt
                df['Pred'] = pred

                df['Probs'] = probabilities
                df = df[
                    ['name_microbe', 'name_disease', 'GT', 'Pred', 'Probs', 'id_microbe', 'id_disease', 'cui_microbe',
                     'cui_disease']]
                all_preds.append(df)

            all_preds = pd.concat(all_preds, axis=0).reset_index(drop=True)
            self.experiment.log_table('Test_predictions_{}.csv'.format(epoch), all_preds, headers=True)




    def train_classifier(self, data, strength_th=0, add_node_features=False, use_otherDBs_mask=True,
                         use_negative_examples=True, binary_classification=False, add_homogenous_edges=True,
                         network_size=32, num_neighbors=[5,5,5,5], batch_size=32, inner_product_dec=False,
                         learning_rate=0.01, add_homogenous_edges_embedding=True, negatives_ratio=0.5,
                       embedding='no', embedding_epochs=10, walk_length_emb=15, context_size_emb=10,
                       walks_per_node_emb =20,
                       num_negative_samples_emb=20, **kwargs):

        # Create the hetero_data
        dataset, masks = self.dataset_creator.create_dataset(data, strength_th=strength_th, negatives_ratio=negatives_ratio,
                                      training_type='classifier', use_negatives_examples=use_negative_examples,
                                                             binary_classification=binary_classification,
                                                             add_homogenous_edges=add_homogenous_edges,
                                                             add_homogenous_edges_embedding=add_homogenous_edges_embedding,
                                                             embedding=embedding, embedding_epochs=embedding_epochs,
                                                             walk_length_emb=walk_length_emb, context_size_emb=context_size_emb,
                                                             walks_per_node_emb=walks_per_node_emb,
                                                             num_negative_samples_emb=num_negative_samples_emb)

        # Split dataset
        train_dataset, val_dataset, test_dataset = self.split_dataset(dataset, edge_type=('microbes', 'strengths', 'diseases'),
                                                                      use_otherDBs_mask=use_otherDBs_mask, masks=masks)

        # Preprocessing
        train_dataset, train_dataloader = self.dataset_preprocessing(train_dataset
                                                                     ,add_homogenous_edges=add_homogenous_edges,
                                                                     num_neighbors=num_neighbors, batch_size=batch_size)
        val_dataset, val_dataloader = self.dataset_preprocessing(val_dataset
                                                                 ,add_homogenous_edges=add_homogenous_edges,
                                                                     num_neighbors=num_neighbors, batch_size=batch_size)
        test_dataset, test_dataloader = self.dataset_preprocessing(test_dataset
                                                                   ,add_homogenous_edges=add_homogenous_edges,
                                                                     num_neighbors=num_neighbors, batch_size=batch_size)



        # Defining the model
        if use_negative_examples and not binary_classification:
            model = Model(hidden_channels=network_size, data=train_dataset, out_features=3).to(device).float()
        else:
            model = Model(hidden_channels=network_size, data=train_dataset, out_features=2).to(device).float()

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        #optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.001)
        criterion = nn.CrossEntropyLoss()

        # Training
        for epoch in range(300):
            train_loss = []
            train_y_binary = []
            train_out_binary = []

            val_loss = []
            val_y_binary = []
            val_out_binary = []

            test_loss = []
            test_y_binary = []
            test_out_binary = []

            model.train()
            for batch_idx, batch in enumerate(tqdm(train_dataloader)):
                optimizer.zero_grad()
                batch = batch.to(device)

                out = model(
                    batch.x_dict,
                    batch.edge_index_dict,
                    batch['microbes', 'diseases'].edge_label_index,
                )
                y = batch['microbes', 'diseases'].edge_label

                loss = criterion(out, y)

               
                loss.backward()
                optimizer.step()
                torch.cuda.empty_cache()

                # Classification_labels
                train_out_binary.append(out.argmax(dim=-1).detach().cpu().numpy().squeeze())
                train_y_binary.append(y.cpu().numpy().squeeze())
                train_loss.append(loss.item())

            train_loss = np.array(train_loss).mean()

            train_y_binary = np.concatenate(train_y_binary)
            train_out_binary = np.concatenate(train_out_binary)

            train_accuracy = accuracy_score(train_y_binary, train_out_binary)
            train_f1 = f1_score(train_y_binary, train_out_binary, average='weighted')
            train_precision = precision_score(train_y_binary, train_out_binary, average='weighted')
            train_recall = recall_score(train_y_binary, train_out_binary, average='weighted')

            print('Epoch {}: Train Loss: {} | Acc: {} | f1: {} | Precision: {} | Recall: {}'.format(epoch, train_loss,
                                                                                                    train_accuracy, train_f1,
                                                                                                    train_precision, train_recall))

            self.experiment.log_metrics({'train_loss': train_loss, 'train_accuracy': train_accuracy,
                                         'train_f1': train_f1, 'train_precision': train_precision,
                                         'train_recall': train_recall}, epoch=epoch)

            # Validation
            with torch.no_grad():
                model.eval()
                for batch_idx, batch in enumerate(tqdm(val_dataloader)):
                    batch = batch.to(device)

                    out = model(
                        batch.x_dict,
                        batch.edge_index_dict,
                        batch['microbes', 'diseases'].edge_label_index,
                    )
                    y = batch['microbes', 'diseases'].edge_label

                    loss = criterion(out, y)



                    # Classification_labels
                    val_out_binary.append(out.argmax(dim=-1).detach().cpu().numpy().squeeze())
                    val_y_binary.append(y.cpu().numpy().squeeze())
                    val_loss.append(loss.item())

            val_loss = np.array(val_loss).mean()

            val_y_binary = np.concatenate(val_y_binary)
            val_out_binary = np.concatenate(val_out_binary)

            val_accuracy = accuracy_score(val_y_binary, val_out_binary)
            val_f1 = f1_score(val_y_binary, val_out_binary, average='weighted')
            val_precision = precision_score(val_y_binary, val_out_binary, average='weighted')
            val_recall = recall_score(val_y_binary, val_out_binary, average='weighted')

            print('Epoch {}: val Loss: {} | Acc: {} | f1: {} | Precision: {} | Recall: {}'.format(epoch, val_loss,
                                                                                                    val_accuracy,
                                                                                                    val_f1,
                                                                                                    val_precision,
                                                                                                    val_recall))

            self.experiment.log_metrics({'val_loss': val_loss, 'val_accuracy': val_accuracy,
                                         'val_f1': val_f1, 'val_precision': val_precision,
                                         'val_recall': val_recall}, epoch=epoch)

            if val_loss < self.min_val:
                self.min_val = val_loss
                self.no_better = 0
                print('Best Model!!!!')
                torch.save(model.state_dict(), 'best_{}.pt'.format(self.experiment_name))
            else:
                self.no_better += 1
                if self.no_better > self.patience:
                    print('Stopping by Early Stopping')
                    return 0

            # Testing
            test_loss = []
            test_out_binary = []
            test_y_binary = []
            test_probabilities = []
            with torch.no_grad():
                test_dataset = test_dataset.to(device)
                out = model(
                    test_dataset.x_dict,
                    test_dataset.edge_index_dict,
                    test_dataset['microbes', 'diseases'].edge_index,
                )

                y = test_dataset['microbes', 'diseases'].edge_label

                loss = criterion(out, y)
                test_loss.append(loss.item())
                test_probabilities.append(out.detach().cpu().numpy().squeeze())
                test_out_binary.append(out.argmax(dim=-1).detach().cpu().numpy().squeeze())
                test_y_binary.append(y.cpu().numpy().squeeze())

            test_loss = np.array(test_loss).mean()
            test_y_binary = np.concatenate(test_y_binary)
            test_out_binary = np.concatenate(test_out_binary)
            test_probabilities = np.concatenate(test_probabilities)

            test_accuracy = accuracy_score(test_y_binary, test_out_binary)
            test_f1 = f1_score(test_y_binary, test_out_binary, average='weighted')
            test_precision = precision_score(test_y_binary, test_out_binary, average='weighted')
            test_recall = recall_score(test_y_binary, test_out_binary, average='weighted')

            print('Epoch {}: test Loss: {} | Acc: {} | f1: {} | Precision: {} | Recall: {}'.format(epoch, test_loss,
                                                                                                   test_accuracy,
                                                                                                   test_f1,
                                                                                                   test_precision,
                                                                                                   test_recall))

            self.experiment.log_metrics({'test_loss': test_loss, 'test_accuracy': test_accuracy,
                                         'test_f1': test_f1, 'test_precision': test_precision,
                                         'test_recall': test_recall}, epoch=epoch)

            # Indices
            edge_indices = test_dataset['microbes', 'diseases'].edge_index.cpu().numpy()
            edge_microbes = edge_indices[0, :]
            edge_diseases = edge_indices[1, :]

            all_preds = []
            for i in range(len(edge_microbes)):
                e_microbe = edge_microbes[i]
                e_disease = edge_diseases[i]
                gt = test_y_binary[i]
                pred = test_out_binary[i]
                probabilities = test_probabilities[i]

                e_microbe_df = data['microbes'][data['microbes']['id'] == e_microbe][
                    ['id', 'cui', 'name']].reset_index(drop=True)
                e_microbe_df.columns = ['id_microbe', 'cui_microbe', 'name_microbe']

                e_disease_df = data['diseases'][data['diseases']['id'] == e_disease][
                    ['id', 'cui', 'name']].reset_index(drop=True)
                e_disease_df.columns = ['id_disease', 'cui_disease', 'name_disease']

                # Appended
                df = pd.concat([e_microbe_df, e_disease_df], axis=1)
                df['GT'] = gt
                df['Pred'] = pred

                df['Probs'] = json.dumps([str(elem) for elem in probabilities])
                df = df[
                    ['name_microbe', 'name_disease', 'GT', 'Pred', 'Probs', 'id_microbe', 'id_disease', 'cui_microbe',
                     'cui_disease']]
                all_preds.append(df)

            all_preds = pd.concat(all_preds, axis=0).reset_index(drop=True)
            self.experiment.log_table('Test_predictions_{}.csv'.format(epoch), all_preds, headers=True)



if __name__ == '__main__':
    os.environ['SM_MODEL_DIR'] = str('results/')
    os.environ['SM_CHANNEL_TRAINING'] = str('../data/')

    seed = 2
    experiment_name = 'GNN2_sentence_{}_full'.format(seed)

    hyperparameters = {'seed': seed, 'task': 'classification', 'experiment_name': experiment_name,
                       'network_size':32, 'inner_product_dec':False,
                       'add_node_features':False, 'use_otherDBs_mask':True,
                             'use_negative_examples':True, 'binary_classification':False,
                             'add_homogenous_edges':True,
                       'num_neighbors': [5, 5, 5, 5], 'batch_size': 32, 'negatives_ratio': 0.5,
                       'learning_rate': 0.01, 'add_homogenous_edges_embedding': True,
                       'embedding': 'sentence', 'embedding_epochs': 10,
                       'walk_length_emb': 10, 'context_size_emb':7, 'walks_per_node_emb':10,
                       'num_negative_samples_emb':10} # metapath2vec


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    folder = os.environ['SM_MODEL_DIR']
    data_folder = os.environ['SM_CHANNEL_TRAINING']

    seed = hyperparameters['seed']

    sentence_encoder = SentenceTransformer('all-MiniLM-L12-v2')


    # Other dbs codifier
    amadis = pd.read_csv(os.path.join(data_folder, 'other_dbs/AMADIS/AMADIS_correct.csv'))
    gmdad = pd.read_csv(os.path.join(data_folder, 'other_dbs/GMDAD/GMDAD_correct.csv'))
    hmdad = pd.read_csv(os.path.join(data_folder, 'other_dbs/HMDAD/HMDAD_correct.csv'))
    disbiome = pd.read_csv(os.path.join(data_folder, 'other_dbs/Disbiome/Disbiome_correct.csv'))
    original = pd.read_csv(os.path.join(data_folder, 'other_dbs/Original/Original_correct.csv'))

    with open(os.path.join(data_folder, 'diseases_cui_id.json'), 'r') as f:
        diseases_dict = json.load(f)

    with open(os.path.join(data_folder, 'microbes_cui_id.json'), 'r') as f:
        microbes_dict = json.load(f)

    databases = [amadis, gmdad, hmdad, disbiome, original]
    codifier = OtherDBsCodifier(databases, diseases_dict, microbes_dict)
    intersections = codifier.process()


    # Load the relevant files
    microbes = pd.read_csv(os.path.join(data_folder,'microbes.csv'))
    diseases = pd.read_csv(os.path.join(data_folder,'diseases.csv'))
    strengths = pd.read_csv(os.path.join(data_folder,'strengths.csv'))
    diseases_genealogy = pd.read_csv(os.path.join(data_folder,'diseases_genealogy.csv'))
    microbes_genealogy = pd.read_csv(os.path.join(data_folder,'microbes_genealogy.csv'))

    data = {'microbes': microbes, 'diseases': diseases, 'strengths': strengths,
            'diseases_genealogy': diseases_genealogy, 'microbes_genealogy': microbes_genealogy}


    # Dataset Creator
    dataset_creator = DatasetCreator(intersections, sentence_encoder, seed=seed)


    # Trainer
    trainer = Trainer(dataset_creator, folder=folder, experiment_name=hyperparameters['experiment_name'],
                      hyperparameters=hyperparameters)


    if hyperparameters['task'] == 'classification':
        trainer.train_classifier(data, **hyperparameters)
    else:
        trainer.train_regressor(data, **hyperparameters)