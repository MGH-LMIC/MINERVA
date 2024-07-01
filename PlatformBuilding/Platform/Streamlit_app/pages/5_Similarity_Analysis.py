import streamlit as st
import pandas as pd
from streamlit_agraph import agraph, Node, Edge, Config
from graph_queries import GraphQueries
import numpy as np
import plotly.graph_objects as go
import copy
import pickle
import plotly.express as px
from ml_queries import MLQueries


querier = GraphQueries()
ml_querier = MLQueries()

st.set_page_config(page_title="Similarity Analysis", page_icon="🧐",  layout='wide')

st.markdown("# Similarity Analysis 🧐")
st.sidebar.header("Explore similar entities in our Graph")
st.sidebar.write(
    """Here you will be able to analyze families of Microbes and Diseases with different types of embeddings"""
)

st.write("--------------------------------------------------")


# Cluster Analysis
st.markdown("## Cluster Analysis")

st.markdown("### Microbe Clusters")
col1, col2 = st.columns(2)
with col1:
    st.markdown("##### Select Embedding Method")
    embedding_type_microbe = st.selectbox(label='hola', options=['FastRP', 'Node2Vec', 'Graph-Sage', 'MetaPath2vec'], index=0, label_visibility='hidden', key=0)
    if embedding_type_microbe == 'FastRP':
        embedding_file_microbe = 'fastRP_embeddings_microbe.pkl'
        kmeans_propery_microbe = 'fastRP-kmeans'
    elif embedding_type_microbe == 'Node2Vec':
        embedding_file_microbe = 'node2vec_embeddings_microbe.pkl'
        kmeans_propery_microbe = 'node2vec-kmeans'
    elif embedding_type_microbe == 'Graph-Sage':
        embedding_file_microbe = 'sage_embeddings_microbe.pkl'
        kmeans_propery_microbe = 'sage-kmeans'
    else:
        embedding_file_microbe = 'metapath_embeddings_microbe.pkl'
        kmeans_propery_microbe = 'metapath-kmeans'

    with open('embeddings/' + embedding_file_microbe, 'rb') as f:
        embeddings_vectors_microbe = pickle.load(f)

with col2:
    st.markdown("##### Select Number of Clusters")
    n_clusters_microbe = st.selectbox(label='hola', options=[3, 4, 5, 6], index=0,
                                          label_visibility='hidden', key=1)

    kmeans_propery_microbe = kmeans_propery_microbe + '{}'.format(n_clusters_microbe)
    clusters_membership = ml_querier.get_clusters(kmeans_propery_microbe, label='Microbe')
    clusters_membership.index = clusters_membership['cui']
    clusters_membership = clusters_membership[['cluster']]

    embeddings_vectors_microbe.index = embeddings_vectors_microbe['cui']
    embeddings_vectors_microbe_cat = pd.concat([embeddings_vectors_microbe, clusters_membership], axis=1)


col1, col2 = st.columns(2)
with col1:

    embeddings_vectors_microbe_cat_table = copy.deepcopy(embeddings_vectors_microbe_cat)[['cui', 'name', 'cluster']]
    embeddings_vectors_microbe_cat_table.columns = ['CUI', 'Name', 'Cluster']
    embeddings_vectors_microbe_cat.index = embeddings_vectors_microbe_cat_table['CUI']
    del embeddings_vectors_microbe_cat_table['CUI']
    embeddings_vectors_microbe_cat_table = embeddings_vectors_microbe_cat_table.sort_values(by='Cluster')
    groups = embeddings_vectors_microbe_cat_table.groupby('Cluster')

    #st.markdown("##### Inspect Cluster N°")
    inspect_cluster_microbe = st.selectbox(label='Inspect Cluster', options=[i for i in range(n_clusters_microbe)], index=0,
                                     key=2)

    if inspect_cluster_microbe not in list(groups.groups.keys()):
        st.markdown("Cluster {} Failed to converge".format(inspect_cluster_microbe))

    else:
        st.dataframe(groups.get_group(inspect_cluster_microbe), use_container_width=True, height=300)


with col2:
    all_names = embeddings_vectors_microbe_cat['name']

    colors = embeddings_vectors_microbe_cat['cluster'].values.reshape(-1, 1)
    sizes = np.ones((embeddings_vectors_microbe_cat.shape[0], 1))*15

    embeddings_vectors_microbe_cat = np.concatenate([embeddings_vectors_microbe_cat[['Dim1', 'Dim2']].values, colors, sizes], axis=1)
    embeddings_vectors_microbe_cat = pd.DataFrame(embeddings_vectors_microbe_cat, columns=['Dim1', 'Dim2', 'color', 'size'], index=all_names)
    embeddings_vectors_microbe_cat = embeddings_vectors_microbe_cat.astype(float)

    fig = go.Figure()  # Create a Figure object directly
    for color in embeddings_vectors_microbe_cat['color'].unique():
        filtered_df = embeddings_vectors_microbe_cat[embeddings_vectors_microbe_cat['color'] == color]
        if filtered_df.shape[0] == 0:
            continue
        if filtered_df.shape[0] > 30:
            filtered_df = filtered_df.sample(frac=0.05, random_state=0)  # Just show a 10% of the entities

        fig.add_trace(go.Scatter(
            x=filtered_df['Dim1'],
            y=filtered_df['Dim2'],
            mode='markers',
            hovertemplate='<b>%{text}</b><br><br>Dim1: %{x}<br>Dim2: %{y}<extra></extra>',
            text=filtered_df.index,  # Add 'name' for hover text
            marker=dict(size=15,
                        line=dict(width=1, color='white')),
            name=int(color)  # This sets the legend label
        ))

    # --- Layout and Styling ---

    fig.update_layout(
        xaxis_title="Dim1",
        yaxis_title="Dim2",
        legend_title="Cluster",
        showlegend=True
    )

    # fig = px.scatter(embeddings_vectors_microbe_cat.reset_index(inplace=False), x="Dim1", y="Dim2", color='color', size='size',
    #                  hover_data=['name'], color_continuous_scale=px.colors.qualitative.Plotly)
    fig.update_layout(coloraxis_showscale=False)
    fig.update_traces(marker=dict(
        line=dict(width=1,
                  color='white')))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("<h6 style='text-align: center; color: grey;'>Only a fraction of the Microbes are displayed to save resources</h6>", unsafe_allow_html=True)



st.markdown("### Disease Clusters")
col1, col2 = st.columns(2)
with col1:
    st.markdown("##### Select Embedding Method")
    embedding_type_disease = st.selectbox(label='hola', options=['FastRP', 'Node2Vec', 'Graph-Sage', 'MetaPath2vec'], index=0, label_visibility='hidden', key=3)
    if embedding_type_disease == 'FastRP':
        embedding_file_disease = 'fastRP_embeddings_disease.pkl'
        kmeans_propery_disease = 'fastRP-kmeans'
    elif embedding_type_disease == 'Node2Vec':
        embedding_file_disease = 'node2vec_embeddings_disease.pkl'
        kmeans_propery_disease = 'node2vec-kmeans'
    elif embedding_type_disease == 'Graph-Sage':
        embedding_file_disease = 'sage_embeddings_disease.pkl'
        kmeans_propery_disease = 'sage-kmeans'
    else:
        embedding_file_disease = 'metapath_embeddings_disease.pkl'
        kmeans_propery_disease = 'metapath-kmeans'

    with open('embeddings/' + embedding_file_disease, 'rb') as f:
        embeddings_vectors_disease = pickle.load(f)

with col2:
    st.markdown("##### Select Number of Clusters")
    n_clusters_disease = st.selectbox(label='hola', options=[3, 4, 5, 6], index=0,
                                          label_visibility='hidden', key=4)

    kmeans_propery_disease = kmeans_propery_disease + '{}'.format(n_clusters_disease)
    clusters_membership_disease = ml_querier.get_clusters(kmeans_propery_disease, label='Disease')
    clusters_membership_disease.index = clusters_membership_disease['cui']
    clusters_membership_disease = clusters_membership_disease[['cluster']]

    embeddings_vectors_disease.index = embeddings_vectors_disease['cui']
    embeddings_vectors_disease_cat = pd.concat([embeddings_vectors_disease, clusters_membership_disease], axis=1)


col1, col2 = st.columns(2)
with col1:

    embeddings_vectors_disease_cat_table = copy.deepcopy(embeddings_vectors_disease_cat)[['cui', 'name', 'cluster']]
    embeddings_vectors_disease_cat_table.columns = ['CUI', 'Name', 'Cluster']
    embeddings_vectors_disease_cat.index = embeddings_vectors_disease_cat_table['CUI']
    del embeddings_vectors_disease_cat_table['CUI']
    embeddings_vectors_disease_cat_table = embeddings_vectors_disease_cat_table.sort_values(by='Cluster')
    groups = embeddings_vectors_disease_cat_table.groupby('Cluster')

    #st.markdown("##### Inspect Cluster N°")
    inspect_cluster_disease = st.selectbox(label='Inspect Cluster', options=[i for i in range(n_clusters_disease)], index=0, key=5)

    if inspect_cluster_disease not in list(groups.groups.keys()):
        st.markdown("Cluster {} Failed to converge".format(inspect_cluster_disease))

    else:
        st.dataframe(groups.get_group(inspect_cluster_disease), use_container_width=True, height=300)


with col2:
    all_names = embeddings_vectors_disease_cat['name']

    colors = embeddings_vectors_disease_cat['cluster'].values.reshape(-1, 1)
    sizes = np.ones((embeddings_vectors_disease_cat.shape[0], 1))*15

    embeddings_vectors_disease_cat = np.concatenate([embeddings_vectors_disease_cat[['Dim1', 'Dim2']].values, colors, sizes], axis=1)
    embeddings_vectors_disease_cat = pd.DataFrame(embeddings_vectors_disease_cat, columns=['Dim1', 'Dim2', 'color', 'size'], index=all_names)
    embeddings_vectors_disease_cat = embeddings_vectors_disease_cat.astype(float)
    # fig = px.scatter(embeddings_vectors_disease_cat.reset_index(inplace=False), x="Dim1", y="Dim2", color='color', size='size',
    #                  hover_data=['name'], color_continuous_scale=px.colors.qualitative.Plotly)
    # fig.update_layout(coloraxis_showscale=False)
    # fig.update_traces(marker=dict(
    #     line=dict(width=1,
    #               color='white')))
    # st.plotly_chart(fig, use_container_width=True)

    fig = go.Figure()  # Create a Figure object directly
    for color in embeddings_vectors_disease_cat['color'].unique():
        filtered_df = embeddings_vectors_disease_cat[embeddings_vectors_disease_cat['color'] == color]

        if filtered_df.shape[0] == 0:
            continue
        if filtered_df.shape[0] > 30:
            filtered_df = filtered_df.sample(frac=0.05, random_state=0) # Just show a 10% of the entities
        fig.add_trace(go.Scatter(
            x=filtered_df['Dim1'],
            y=filtered_df['Dim2'],
            mode='markers',
            hovertemplate='<b>%{text}</b><br><br>Dim1: %{x}<br>Dim2: %{y}<extra></extra>',
            text=filtered_df.index,  # Add 'name' for hover text
            marker=dict(size=15,
                        line=dict(width=1, color='white')),
            name=int(color)  # This sets the legend label
        ))

    # --- Layout and Styling ---

    fig.update_layout(
        xaxis_title="Dim1",
        yaxis_title="Dim2",
        legend_title="Cluster",
        showlegend=True
    )

    # fig = px.scatter(embeddings_vectors_microbe_cat.reset_index(inplace=False), x="Dim1", y="Dim2", color='color', size='size',
    #                  hover_data=['name'], color_continuous_scale=px.colors.qualitative.Plotly)
    fig.update_layout(coloraxis_showscale=False)
    fig.update_traces(marker=dict(
        line=dict(width=1,
                  color='white')))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("<h6 style='text-align: center; color: grey;'>Only a fraction of the Diseases are displayed to save resources</h6>", unsafe_allow_html=True)





# Graph connections
