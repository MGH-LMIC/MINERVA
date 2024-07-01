import numpy as np
import pandas as pd
from torch_geometric.nn import Node2Vec
import torch
from dataset_creator import DatasetCreator
import json
from torch_geometric.transforms import Compose, ToUndirected, RandomLinkSplit, RemoveIsolatedNodes, RemoveDuplicatedEdges
from torch_geometric.nn import MetaPath2Vec
import matplotlib.pyplot as plt
import umap
import umap.plot
import seaborn as sns
from matplotlib.colors import ListedColormap
import mplcursors
from sklearn.cluster import KMeans
from py2neo import Graph, NodeMatcher



class CreateEmbeddings:
    def __init__(self, dataset_creator):
        self.dataset_creator = dataset_creator
        self.clusters = KMeans(n_clusters=2)
        self.graph = Graph(host="0.0.0.0", port="7687", auth=('neo4j', 'your_password'))


    def train_node2vec(self, data, include_homogeneous_relations=False, epochs=500):
        # Create the hetero_data
        dataset, masks = self.dataset_creator.create_dataset(data, strength_th=0,
                                                             add_node_features=False,
                                                             training_type='regressor',
                                                             use_negatives_examples=False)
        dataset = ToUndirected()(dataset)

        if include_homogeneous_relations:

            # Since it doesn't make any distinction between different kind of nodes Ids must be normalized
            microbes_max_id = microbes['id'].values[-1]

            # Hetero edges
            hetero_edges = dataset['microbes', 'diseases'].edge_index
            hetero_edges[1, :] = hetero_edges[1, :] + microbes_max_id # All diseases ids are increased in one

            hetero_edges_rev = dataset['diseases', 'microbes'].edge_index
            hetero_edges_rev[0, :] = hetero_edges_rev[0, :] + microbes_max_id # All diseases ids are increased in one

            hetero_edges = torch.concat([hetero_edges, hetero_edges_rev], dim=1)

            # Homo edges
            homo_edges_microbes = dataset['microbes', 'microbes'].edge_index
            homo_edges_diseases = dataset['diseases', 'diseases'].edge_index + microbes_max_id

            hetero_edges = torch.cat([hetero_edges, homo_edges_microbes, homo_edges_diseases], dim=1)
        else:

            hetero_edges = dataset['microbes', 'diseases'].edge_index

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
            walks_per_node=10,
            walk_length=20,
            context_size=10,
            p=1.0,
            q=1.0,
            num_negative_samples=5,
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
            print('Epoch {}: Loss: {}'.format(epoch, total_loss))

        # Embeddings
        z = model()  # Full node-level embeddings.
        z_microbes = z[:microbes_max_id]
        z_diseases = z[microbes_max_id:]
        print(z_microbes.shape, z_diseases.shape)

        self.plot_umap(z_microbes, z_diseases)
        #z = model(torch.tensor([0, 1, 2]))  # Embeddings of first three nodes.


    def train_metapath2vec(self, data, epochs=500, include_only_connected=True):
        # Create the hetero_data
        dataset, masks = self.dataset_creator.create_dataset(data, strength_th=0,
                                                             training_type='regressor',
                                                             use_negatives_examples=False)

        dataset = ToUndirected()(dataset)
        #dataset = RemoveIsolatedNodes()(dataset)
        print(dataset)
        hetero_edges = dataset['microbes', 'diseases'].edge_index

        metapath = [
            ('microbes', 'strengths', 'diseases'),
            ('diseases', 'parent_d', 'diseases'),
            ('diseases', 'rev_strengths', 'microbes'),
            ('microbes', 'parent_m', 'microbes'),
        ]


        model = MetaPath2Vec(dataset.edge_index_dict, embedding_dim=64,
                             metapath=metapath, walk_length=10, context_size=7,
                             walks_per_node=10, num_negative_samples=10,
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
            print('Epoch {}: Loss: {}'.format(epoch, total_loss))

        # Embeddings
        z_microbes = model(node_type='microbes')  # Full node-level embeddings.
        z_diseases = model(node_type='diseases')


        print(z_microbes.shape, z_diseases.shape)

        diseases_cui = data['diseases']['cui'].values
        microbes_cui = data['microbes']['cui'].values

        self.upload_to_neo(z_diseases.detach().cpu().numpy(), diseases_cui, z_microbes.detach().cpu().numpy(), microbes_cui)


    def upload_to_neo(self, z_diseases, diseases_cui, z_microbes, microbes_cui):
        print('Uploading Diseases Embeddings to Neo')
        for i in range(len(z_diseases)):
            d_emb = z_diseases[i].tolist()
            d_cui = diseases_cui[i]

            if i % 200 == 0:
                print('{}|{}'.format(i, len(z_diseases)))

            query = """
            MATCH (n:Disease)
                WHERE n.cui = '{}'
                SET n.`metapath-embedding` = {}
            """.format(d_cui, d_emb)
            self.graph.run(query)

        print('Uploading Microbes Embeddings to Neo')
        for i in range(len(z_microbes)):
            m_emb = z_microbes[i].tolist()
            m_cui = microbes_cui[i]

            if i % 200 == 0:
                print('{}|{}'.format(i, len(z_microbes)))

            query = """
                   MATCH (n:Microbe)
                       WHERE n.cui = '{}'
                       SET n.`metapath-embedding` = {}
                   """.format(m_cui, m_emb)
            self.graph.run(query)

    def plot_umap(self, z_microbes, z_diseases, microbes_info, diseases_info, edges=[], seed=0, n_neighbors=15,
                  metric="euclidean", min_dist=0.1, connected_only=True, include_arrows=False):
        reducer = umap.UMAP(n_components=2, random_state=seed, n_neighbors=n_neighbors, metric=metric, min_dist=min_dist)

        z_microbes = z_microbes.detach().cpu().numpy()
        z_diseases = z_diseases.detach().cpu().numpy()

        z = np.concatenate([z_microbes, z_diseases])
        labels = ['Microbes' for i in range(len(z_microbes))] + ['Diseases' for i in range(len(z_diseases))]

        mapper = reducer.fit(z)
        torch.save(mapper, 'mapper.pkl')
        z_trans = mapper.transform(z)
        z_trans_microbes = z_trans[:z_microbes.shape[0]]
        z_trans_diseases = z_trans[z_microbes.shape[0]:]

        m_coords = []
        d_coords = []
        for i in range(edges.shape[1]):
            microbes_idx = edges[0, i]
            diseases_idx = edges[1, i]
            m_coord = z_trans_microbes[microbes_idx]
            d_coord = z_trans_diseases[diseases_idx]
            m_coords.append(m_coord)
            d_coords.append(d_coord)

        if connected_only:
            z_trans_microbes = z_trans_microbes[edges[0, :]]
            z_trans_diseases = z_trans_diseases[edges[1, :]]

        # Nature-inspired style adjustments
        plt.figure(figsize=(21, 10))  # Slightly smaller size
        sns.set_style("ticks")  # Minimalist style
        plt.rcParams.update({
            "font.family": "serif",  # Serif font (e.g., Times New Roman)
            "font.size": 12,
            "axes.linewidth": 0.8,  # Thinner axes
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
        })

        if include_arrows:
            for i in range(len(m_coords)):
                m_coord = m_coords[i]
                d_coord = d_coords[i]

                plt.arrow(m_coord[0], m_coord[1], d_coord[0] - m_coord[0], d_coord[1] - m_coord[1],
                          color="gray", alpha=0.3, width=0.001, head_width=0, length_includes_head=False)

        # Scatter plot with muted colors
        scatter2 = plt.scatter(z_trans_diseases[:, 0], z_trans_diseases[:, 1],
                               c="lightcoral", s=30, alpha=0.7, edgecolors="none", label="Diseases")



        scatter1 = plt.scatter(z_trans_microbes[:, 0], z_trans_microbes[:, 1],
                               c="yellowgreen", s=30, alpha=0.7, edgecolors="none", label="Microbes")


        # Axis labels and title
        plt.xlabel("Feature 1", fontsize=12)
        plt.ylabel("Feature 2", fontsize=12)
        # No title for a cleaner look (optional)



        # Legend adjustments
        legend = plt.legend(handles=[scatter1, scatter2], title="", fontsize=10,
                            handletextpad=0.2, borderpad=0.3, loc="upper right")
        plt.gca().add_artist(legend)

        # Remove top and right spines for a cleaner look
        sns.despine()

        # Optionally, use sciplotlib for further refinement
        try:
            import sciplotlib.style as splstyle
            import sciplotlib.polish as splpolish
            with plt.style.context(splstyle.get_style("nature-reviews")):
                fig, ax = plt.gcf(), plt.gca()
                fig, ax = splpolish.set_bounds(fig, ax)  # Adjust axis limits
        except ImportError:
            pass  # sciplotlib not installed

        if microbes_info.shape[0] > 0:
            print(microbes_info)
            microbes_names = microbes_info['name'].values.squeeze()
            diseases_names = diseases_info['name'].values.squeeze()
            microbes_cui = microbes_info['cui'].values.squeeze()
            diseases_cui = diseases_info['cui'].values.squeeze()

            if connected_only:
                microbes_names = microbes_names[edges[0, :]]
                diseases_names = diseases_names[edges[1, :]]
                microbes_cui = microbes_cui[edges[0, :]]
                diseases_cui = diseases_cui[edges[1, :]]

            cursor = mplcursors.cursor([scatter1, scatter2], hover=True)

            @cursor.connect("add")
            def on_add(sel):
                i = sel.target.index
                if sel.artist == scatter1:
                    cui = microbes_cui[i]
                    name = microbes_names[i]
                else:  # sel.artist == scatter2
                    cui = diseases_cui[i]
                    name = diseases_names[i]


                sel.annotation.set_text(
                    f"Point {i}\n"
                    f"Cui: {cui}\n"
                    f"Name: {name}"
                )


        # Show the plot
        plt.tight_layout()
        plt.savefig('umap_arrows.png')
        plt.savefig('umap_dpi250_arrows.png', dpi=250)
        plt.show()


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')

    seed = 0


    # Load the relevant files
    microbes = pd.read_csv('data/microbes.csv')
    diseases = pd.read_csv('data/diseases.csv')
    strengths = pd.read_csv('data/strengths.csv')
    diseases_genealogy = pd.read_csv('data/diseases_genealogy.csv')
    microbes_genealogy = pd.read_csv('data/microbes_genealogy.csv')

    data = {'microbes': microbes, 'diseases': diseases, 'strengths': strengths,
            'diseases_genealogy': diseases_genealogy, 'microbes_genealogy': microbes_genealogy}

    # Dataset Creator
    intersections = []
    dataset_creator = DatasetCreator(intersections, sentence_encoder=[], seed=seed)


    create_embeddings = CreateEmbeddings(dataset_creator)

    #create_embeddings.train_node2vec(data, epochs=5)
    create_embeddings.train_metapath2vec(data, epochs=10, include_only_connected=False)

    