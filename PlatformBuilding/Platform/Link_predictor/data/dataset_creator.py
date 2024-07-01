import pandas as pd
import numpy as np
import copy
import torch
from torch_geometric.data import HeteroData
from torch_geometric.nn import MetaPath2Vec, Node2Vec
from torch_geometric.transforms import Compose, ToUndirected


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DatasetCreator:
    def __init__(self, intersections, sentence_encoder, seed=0):
        self.intersections = intersections
        self.seed = seed
        self.sentence_encoder = sentence_encoder

    def preprocess_strengths(self, strengths, strength_th=0, training_type='regressor', train_val_ratio=0.8,
                             use_negatives_examples=False, binary_classification=False, negatives_ratio=0.5):
        th_pos = strength_th
        th_neg = -strength_th

        # Filter by th_first
        if th_pos == 0:
            strengths_positives = strengths[strengths['strength_raw'] > th_pos]
            strengths_negatives = strengths[strengths['strength_raw'] < th_neg]
            strengths_unk = strengths[strengths['strength_raw'] == th_pos]

        else:
            strengths_positives = strengths[strengths['strength_raw'] >= th_pos]
            strengths_negatives = strengths[strengths['strength_raw'] < th_neg]

        # Transform to a regression problem
        if training_type == 'regressor':
            strengths_positives['strength_raw'] = 1
            strengths_negatives['strength_raw'] = -1

            strengths = pd.concat([strengths_positives, strengths_negatives], axis=0)
            strengths = strengths.drop_duplicates()

            strengths = strengths.sample(frac=1, random_state=self.seed)
            strengths['id'] = np.array([j for j in range(len(strengths))])
            strengths.index = strengths['id_microbe'].astype(str) + '_' + strengths['id_disease'].astype(str)

        else:
            if use_negatives_examples and not binary_classification:
                strengths_positives['strength_raw'] = 2
                strengths_negatives['strength_raw'] = 1
            elif use_negatives_examples and binary_classification:
                strengths_positives['strength_raw'] = 1
                strengths_negatives['strength_raw'] = 1
            else:
                strengths_positives['strength_raw'] = 1
                strengths_negatives['strength_raw'] = 0

            strengths = pd.concat([strengths_positives, strengths_negatives], axis=0)
            strengths = strengths.drop_duplicates()

            strengths.index = strengths['id_microbe'].astype(str) + '_' + strengths['id_disease'].astype(str)

            if use_negatives_examples:
                #### Augmentations ####
                if binary_classification:
                    n_counter_examples = len(strengths)
                else:
                    n_counter_examples = int(len(strengths)*negatives_ratio)

                id_microbes = np.array(list(copy.deepcopy(strengths['id_microbe'].values.squeeze())))
                id_diseases = np.array(list(copy.deepcopy(strengths['id_disease'].values.squeeze())))

                counter_microbes = []
                counter_diseases = []
                for jj in range(3):
                    np.random.seed(self.seed + jj)
                    np.random.shuffle(id_diseases)
                    np.random.shuffle(id_microbes)

                    counter_microbes += list(id_microbes)
                    counter_diseases += list(id_diseases)

                counter_indices = ['{}_{}'.format(counter_microbes[i], counter_diseases[i]) for i in range(len(counter_microbes))]

                # Not in index
                counter_indices = list(set(counter_indices) - set(list(strengths.index)))[:n_counter_examples]

                # Creating the df
                counter_microbes = np.array([int(elem.split('_')[0]) for elem in counter_indices]).reshape(-1, 1)
                counter_diseases = np.array([int(elem.split('_')[1]) for elem in counter_indices]).reshape(-1, 1)
                counter_strengths = np.array([0 for _ in range(len(counter_indices))]).reshape(-1, 1)

                counter_df = pd.DataFrame(np.concatenate([counter_microbes, counter_diseases, counter_strengths, counter_strengths], axis=1),
                                          index=counter_indices, columns=['id_microbe', 'id_disease', 'strength_raw', 'strength_IF'])

                # Concatenating
                strengths = pd.concat([strengths, counter_df], axis=0)

            strengths = strengths.sample(frac=1, random_state=self.seed)
            strengths['id'] = np.array([j for j in range(len(strengths))])

            strengths = strengths[~strengths.index.duplicated(keep='first')]


        # Getting the testing, training and validation masks

        # Testing mask
        testing_indices = np.array([i for i in range(10000)])
        np.random.seed(self.seed)
        np.random.shuffle(testing_indices)
        testing_mask = strengths.iloc[testing_indices, :]['id'].values.squeeze()

        # Training val mask
        training_val_mask = np.array(list(set(strengths['id'].values.tolist()) - set(list(testing_mask))))
        training_indices = np.array([i for i in range(len(training_val_mask))])
        np.random.seed(self.seed)
        np.random.shuffle(training_indices)

        training_mask = training_val_mask[training_indices[:int(len(training_indices)*train_val_ratio)]]
        val_mask = training_val_mask[training_indices[int(len(training_indices)*train_val_ratio):]]


        # Calculating indices and labels
        edge_index = strengths[['id_microbe', 'id_disease']].values.T

        edge_label = strengths['strength_raw'].values.squeeze()

        return edge_index, edge_label, strengths, [training_mask, val_mask, testing_mask]

    def process_parents(self, parents):
        edge_index = parents[['id_orig', 'id_target']].values.T
        edge_label = np.ones(shape=(len(parents), 1))

        return edge_index, edge_label, parents

    def train_node2vec(self, dataset, masks, include_homogeneous_relations=False, epochs=500,
                       walk_length_emb=15, context_size_emb=10,
                       walks_per_node_emb=20,
                       num_negative_samples_emb=20):
        # Create the hetero_data

        my_dataset = copy.deepcopy(dataset)
        training_mask = masks[0]
        val_mask = masks[1]
        my_dataset['microbes', 'diseases'].edge_index = my_dataset['microbes', 'diseases'].edge_index[:, training_mask]
        my_dataset = ToUndirected()(my_dataset)
        print(my_dataset)

        if include_homogeneous_relations:

            # Since it doesn't make any distinction between different kind of nodes Ids must be normalized
            microbes_max_id = my_dataset['microbes'].x.shape[0]

            # Hetero edges
            hetero_edges = my_dataset['microbes', 'diseases'].edge_index
            hetero_edges[1, :] = hetero_edges[1, :] + microbes_max_id # All diseases ids are increased in one

            hetero_edges_rev = my_dataset['diseases', 'microbes'].edge_index
            hetero_edges_rev[0, :] = hetero_edges_rev[0, :] + microbes_max_id # All diseases ids are increased in one

            hetero_edges = torch.concat([hetero_edges, hetero_edges_rev], dim=1)

            # Homo edges
            homo_edges_microbes = my_dataset['microbes', 'microbes'].edge_index
            homo_edges_diseases = my_dataset['diseases', 'diseases'].edge_index + microbes_max_id

            hetero_edges = torch.cat([hetero_edges, homo_edges_microbes, homo_edges_diseases], dim=1)
        else:

            hetero_edges = my_dataset['microbes', 'diseases'].edge_index

            # Relabel microbes (row 0)
            _, unique_microbe_indices = torch.unique(hetero_edges[0], return_inverse=True)

            new_edge_index = hetero_edges.clone()
            new_edge_index[0] = unique_microbe_indices
            microbes_max_id = unique_microbe_indices.max()

            # Relabel diseases (row 1)
            _, unique_disease_indices = torch.unique(new_edge_index[1], return_inverse=True)
            new_edge_index[1] = unique_disease_indices + microbes_max_id

            # Include reverse_edges
            hetero_edges_rev = new_edge_index[[1,0], :]
            hetero_edges = new_edge_index
            hetero_edges = torch.cat([hetero_edges, hetero_edges_rev], dim=1)


        model = Node2Vec(
            hetero_edges,
            embedding_dim=128,
            walks_per_node=walks_per_node_emb,
            walk_length=walk_length_emb,
            context_size=context_size_emb,
            p=1.0,
            q=1.0,
            num_negative_samples=num_negative_samples_emb,
        ).to(device)

        optimizer = torch.optim.Adam(list(model.parameters()), lr=0.01)

        loader = model.loader(batch_size=128, shuffle=True, num_workers=4)

        # Training the model
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            for pos_rw, neg_rw in loader:
                optimizer.zero_grad()
                loss = model.loss(pos_rw.to(device), neg_rw.to(device))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            total_loss / len(loader)
            print('Node2vec Epoch {}: Loss: {}'.format(epoch, total_loss))

        # Embeddings
        z = model()  # Full node-level embeddings.
        z_microbes = z[:microbes_max_id]
        z_diseases = z[microbes_max_id:]

        return z_microbes, z_diseases


    def train_metapath2vec(self, dataset, mask, epochs=500, include_homogeneous_relations=False,
                           walk_length_emb=15, context_size_emb=10,
                           walks_per_node_emb=20,
                           num_negative_samples_emb=20):
        # Create the hetero_data
        my_dataset = copy.deepcopy(dataset)
        training_mask = mask[0]
        my_dataset['microbes', 'diseases'].edge_index = my_dataset['microbes', 'diseases'].edge_index[:, training_mask]

        my_dataset = ToUndirected()(my_dataset)

        if include_homogeneous_relations:
            metapath = [
                ('microbes', 'strengths', 'diseases'),
                ('diseases', 'rev_strengths', 'microbes'),
            ]
        else:
            metapath = [
                ('microbes', 'strengths', 'diseases'),
                ('diseases', 'parent_d', 'diseases'),
                ('diseases', 'rev_strengths', 'microbes'),
                ('microbes', 'parent_m', 'microbes'),
            ]

        model = MetaPath2Vec(my_dataset.edge_index_dict, embedding_dim=128,
                             metapath=metapath, walk_length=walk_length_emb, context_size=context_size_emb,
                             walks_per_node=walks_per_node_emb, num_negative_samples=num_negative_samples_emb,
                             sparse=True).to(device)

        optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=0.01)
        loader = model.loader(batch_size=128, shuffle=True, num_workers=4)

        # Training the model
        model.train()
        for epoch in range(epochs):
            total_loss = 0
            for pos_rw, neg_rw in loader:
                optimizer.zero_grad()
                loss = model.loss(pos_rw.to(device), neg_rw.to(device))
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            total_loss / len(loader)
            print('Metapath2vec Epoch {}: Loss: {}'.format(epoch, total_loss))

        # Embeddings
        z_microbes = model(node_type='microbes')  # Full node-level embeddings.
        z_diseases = model(node_type='diseases')

        return z_microbes, z_diseases

    def preprocess_nodes(self, microbes, diseases, dataset, masks, embedding='no', embedding_epochs=10,
                         include_homogeneous_relations=False, walk_length_emb=15, context_size_emb=10,
                       walks_per_node_emb=20,
                       num_negative_samples_emb=20):

        if embedding == 'sentence':
            features = []
            for nodes in [microbes, diseases]:
                definitions = nodes['definition'].values.squeeze()
                batch_size = 50
                batch = 0
                all_features = []
                while batch < len(definitions):
                    X = definitions[batch:batch + batch_size]
                    all_features.append(self.sentence_encoder.encode(X))
                    batch += batch_size
                features.append(np.concatenate(all_features, axis=0))
            print(features[0].shape, features[1].shape)
            dataset['microbes'].x = torch.from_numpy(features[0])
            dataset['diseases'].x = torch.from_numpy(features[1])

        elif embedding == 'node2vec':
            z_microbes, z_diseases = self.train_node2vec(dataset, masks,
                                                         include_homogeneous_relations=include_homogeneous_relations,
                                                         epochs=embedding_epochs,
                                                         walk_length_emb=walk_length_emb, context_size_emb=context_size_emb,
                                                         walks_per_node_emb=walks_per_node_emb,
                                                         num_negative_samples_emb=num_negative_samples_emb
                                                         )
            dataset['microbes'].x = z_microbes
            dataset['diseases'].x = z_diseases

        elif embedding == 'metapath2vec':
            z_microbes, z_diseases = self.train_metapath2vec(dataset, masks, epochs=embedding_epochs,
                                                             include_homogeneous_relations=include_homogeneous_relations,
                                                             walk_length_emb=walk_length_emb, context_size_emb=context_size_emb,
                                                             walks_per_node_emb=walks_per_node_emb,
                                                             num_negative_samples_emb=num_negative_samples_emb
                                                             )
            dataset['microbes'].x = z_microbes
            dataset['diseases'].x = z_diseases

        elif embedding == 'autoencoder':
            pass
        else:
            pass

        return dataset

    def create_dataset(self, data, strength_th=0,
                       add_homogenous_edges=True,
                       add_homogenous_edges_embedding=True,
                       embedding='no', embedding_epochs=10,
                       train_val_ratio=0.8,
                       training_type='regression',
                       negatives_ratio=0.5,
                       use_negatives_examples=False, binary_classification=False,
                       walk_length_emb=15, context_size_emb=10,
                       walks_per_node_emb=20,
                       num_negative_samples_emb=20):

        ### Process indices ###
        edge_index, edge_label, strengths, masks = self.preprocess_strengths(data['strengths'], strength_th=strength_th,
                                                                             negatives_ratio=negatives_ratio,
                                                                             train_val_ratio=train_val_ratio,
                                                                             training_type=training_type,
                                                                             use_negatives_examples=use_negatives_examples,
                                                                             binary_classification=binary_classification)

        dg_index, dg_label, diseases_genealogy = self.process_parents(data['diseases_genealogy'])
        mg_index, mg_label, microbes_genealogy = self.process_parents(data['microbes_genealogy'])

        ### Create dataset ###
        dataset = HeteroData()

        # Add nodes
        dataset['microbes'].x = torch.from_numpy(np.eye(len(data['microbes']))).float()
        dataset['diseases'].x = torch.from_numpy(np.eye(len(data['diseases']))).float()

        # Add edges
        dataset['microbes', 'strengths', 'diseases'].edge_index = torch.from_numpy(edge_index)
        dataset['microbes', 'strengths', 'diseases'].edge_label = torch.from_numpy(edge_label)

        dataset['microbes', 'parent_m', 'microbes'].edge_index = torch.from_numpy(mg_index)
        dataset['diseases', 'parent_d', 'diseases'].edge_index = torch.from_numpy(dg_index)

        # transforms = Compose([RemoveIsolatedNodes(), RemoveDuplicatedEdges()])
        # dataset = transforms(dataset)

        ### Process nodes ###
        dataset = self.preprocess_nodes(data['microbes'], data['diseases'], dataset, masks, embedding=embedding,
                                        embedding_epochs=embedding_epochs,
                                        walk_length_emb=walk_length_emb, context_size_emb=context_size_emb,
                                        walks_per_node_emb=walks_per_node_emb,
                                        num_negative_samples_emb=num_negative_samples_emb,
                                        include_homogeneous_relations=add_homogenous_edges_embedding)

        if not add_homogenous_edges:
            del dataset['microbes', 'parent_m', 'microbes']
            del dataset['diseases', 'parent_d', 'diseases']

        return dataset, masks