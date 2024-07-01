import streamlit as st
import pandas as pd
from graph_queries import GraphQueries
from ml_queries import MLQueries
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go

querier = GraphQueries()
ml_querier = MLQueries()

st.set_page_config(page_title="Link Prediction", page_icon="🤖",  layout='wide')

st.markdown("# Link Prediction 🤖")
st.sidebar.header("Link Prediction")
st.sidebar.write(
    """Link prediction of possible new connections"""
)

st.write("--------------------------------------------------")

# Disease Analysis
st.markdown("## Microbiome Analysis with ML 🦠")

col1, col2 = st.columns(2)
with col1:
    st.markdown("##### Select a Microbe")
    microbes = querier.get_all_microbes()
    names = [elem.capitalize() for elem in microbes['name'].values.squeeze()]
    cuis = [elem for elem in microbes['cui'].values.squeeze()]
    names = ['{} ({})'.format(names[i], cuis[i]) for i in range(len(cuis))]
    option = st.selectbox(label='hola', options=names, index=0, label_visibility='hidden', key=0)
    microbe_name = option.split('(C')[0].strip()
    microbe_cui = option.split('(C')[1]
    microbe_cui = 'C' + microbe_cui[:-1]

with col2:
    st.markdown("##### Select Embedding Method")
    embedding_type = st.selectbox(label='None', options=['FastRP', 'Node2vec', 'Graph-Sage', 'MetaPath2vec'], label_visibility='hidden', key=1)

st.write("--------------------------------------------------")

st.markdown("### Nearest Neighbors")
col1, col2 = st.columns(2)

with col1:
    if embedding_type == 'FastRP':
        embedding_property = 'fastRP-embedding'
    elif embedding_type == 'Node2vec':
        embedding_property = 'node2vec-embedding'
    elif embedding_type == 'Graph-Sage':
        embedding_property = 'sage-embedding'
    else:
        embedding_property = 'metapath-embedding'

    st.markdown("""
        <style>
        table {
            font-family: 'Arial', sans-serif; 
            border-collapse: collapse;
            font-size:15px;
            text-align:center;

        }
        th {
            font-weight: bold;
        }
         td {
            text-align:center;
        }
        </style>
        """, unsafe_allow_html=True)

    microbiome_neighs = ml_querier.find_nearest_neighbors(embedding_property=embedding_property, label='Microbe', cui=microbe_cui)
    microbiome_neighs.columns = ['Name', 'Official Name', 'CUI', 'Similarity']
    microbiome_neighs.index = np.array([i + 1 for i in range(len(microbiome_neighs))])
    #st.dataframe(microbiome_neighs, use_container_width=True)
    st.table(microbiome_neighs)

with col2:
    if embedding_type == 'FastRP':
        embedding_filename = 'embeddings/fastRP_embeddings_microbe.pkl'
    elif embedding_type == 'Node2vec':
        embedding_filename = 'embeddings/node2vec_embeddings_microbe.pkl'
    elif embedding_type == 'Graph-Sage':
        embedding_filename = 'embeddings/sage_embeddings_microbe.pkl'
    else:
        embedding_filename = 'embeddings/metapath_embeddings_microbe.pkl'

    with open(embedding_filename, 'rb') as f:
        embeddings_vectors = pickle.load(f)

    print(embeddings_vectors.columns)
    all_names = embeddings_vectors['name']

    target_name = microbe_name.lower()
    target_index = all_names.to_list().index(target_name)
    colors = np.zeros((all_names.shape[0], 1))
    max_col = np.max(colors)
    colors[target_index, :] = max_col + 1

    sizes = np.ones((embeddings_vectors.shape[0], 1))*15
    sizes[target_index, :] = 25

    embeddings_vectors = np.concatenate([embeddings_vectors.values[:, 2:], colors, sizes], axis=1)
    embeddings_vectors = pd.DataFrame(embeddings_vectors, columns=['Dim1', 'Dim2', 'color', 'size'], index=all_names)
    embeddings_vectors = embeddings_vectors.astype(float)


    fig = go.Figure()  # Create a Figure object directly
    for color in embeddings_vectors['color'].unique():
        filtered_df = embeddings_vectors[embeddings_vectors['color'] == color]
        if filtered_df.shape[0] > 1:
            filtered_df = filtered_df.sample(frac=0.1, random_state=0)

        if filtered_df.shape[0] == 1:
            mycol = 'darkgreen'
        else:
            mycol = 'yellowgreen'

        fig.add_trace(go.Scatter(
            x=filtered_df['Dim1'],
            y=filtered_df['Dim2'],
            mode='markers',
            hovertemplate='<b>%{text}</b><br><br>Dim1: %{x}<br>Dim2: %{y}<extra></extra>',
            text=filtered_df.index,  # Add 'name' for hover text
            marker=dict(size=filtered_df['size'].values.squeeze(),
                        color=mycol,
                        line=dict(width=1, color='white')),
        ))

    # --- Layout and Styling ---

    fig.update_layout(
        xaxis_title="Dim1",
        yaxis_title="Dim2",
        legend_title="Cluster",
        showlegend=False
    )

    fig.update_layout(coloraxis_showscale=False)
    fig.update_traces(marker=dict(
        line=dict(width=1,
                  color='white')))
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("<h6 style='text-align: center; color: grey;'>Only a fraction of the Microbes are displayed to save resources</h6>", unsafe_allow_html=True)


st.write("--------------------------------------------------")
st.markdown("### Link Prediction")
col1, col2 = st.columns(2)
positive_links, negative_links = ml_querier.find_predicted_links_microbe(cui=microbe_cui)
if positive_links.shape[0] > 0:
    positive_links.columns = ['Disease', 'Strength', 'CUI']
    positive_links.index = np.array([i + 1 for i in range(len(positive_links))])
if negative_links.shape[0] > 0:
    negative_links.columns = ['Disease', 'Strength', 'CUI']
    negative_links['Strength'] = - negative_links['Strength']
    negative_links.index = np.array([i + 1 for i in range(len(negative_links))])


with col1:
    st.markdown('### Positive Predicted Links')
    st.table(positive_links)

with col2:
    st.markdown('### Negative Predicted Links')
    st.table(negative_links)



st.write("--------------------------------------------------")

# Disease Analysis
st.markdown("## Diseases Analysis with ML 🤢")

col1, col2 = st.columns(2)
with col1:
    st.markdown("##### Select a Disease")
    microbes = querier.get_all_diseases()
    names = [elem.capitalize() for elem in microbes['name'].values.squeeze()]
    cuis = [elem for elem in microbes['cui'].values.squeeze()]
    names = ['{} ({})'.format(names[i], cuis[i]) for i in range(len(cuis))]
    option_disease = st.selectbox(label='hola', options=names, index=0, label_visibility='hidden', key=2)
    disease_name = option_disease.split('(C')[0].strip()
    disease_cui = option_disease.split('(C')[1]
    disease_cui = 'C' + disease_cui[:-1]

with col2:
    st.markdown("##### Select Embedding Method")
    embedding_type_disease = st.selectbox(label='None', options=['FastRP', 'Node2vec', 'Graph-Sage', 'MetaPath2vec'], label_visibility='hidden', key=3)

st.write("--------------------------------------------------")

st.markdown("### Nearest Neighbors")
col1, col2 = st.columns(2)

with col1:
    if embedding_type_disease == 'fastRP':
        embedding_property = 'fastRP-embedding'
    elif embedding_type_disease == 'Node2vec':
        embedding_property = 'node2vec-embedding'
    elif embedding_type_disease == 'Graph-Sage':
        embedding_property = 'sage-embedding'
    else:
        embedding_property = 'metapath-embedding'


    st.markdown("""
        <style>
        table {
            font-family: 'Arial', sans-serif; 
            border-collapse: collapse;
            font-size:15px;
            text-align:center;

        }
        th {
            font-weight: bold;
        }
         td {
            text-align:center;
        }
        </style>
        """, unsafe_allow_html=True)

    microbiome_neighs = ml_querier.find_nearest_neighbors(embedding_property=embedding_property, label='Disease', cui=disease_cui)
    microbiome_neighs.columns = ['Name', 'Official Name', 'CUI', 'Similarity']
    microbiome_neighs.index = np.array([i + 1 for i in range(len(microbiome_neighs))])
    #st.dataframe(microbiome_neighs, use_container_width=True)
    st.table(microbiome_neighs)

with col2:
    if embedding_type_disease == 'FastRP':
        embedding_filename = 'embeddings/fastRP_embeddings_disease.pkl'
    elif embedding_type_disease == 'Node2vec':
        embedding_filename = 'embeddings/node2vec_embeddings_disease.pkl'
    elif embedding_type_disease == 'Graph-Sage':
        embedding_filename = 'embeddings/sage_embeddings_disease.pkl'
    else:
        embedding_filename = 'embeddings/metapath_embeddings_disease.pkl'


    with open(embedding_filename, 'rb') as f:
        embeddings_vectors_disease = pickle.load(f)

    all_names = embeddings_vectors_disease['name']
    target_name = disease_name.lower()
    target_index = all_names.to_list().index(target_name)
    colors = np.zeros((all_names.shape[0], 1))
    max_col = np.max(colors)
    colors[target_index, :] = max_col + 1
    sizes = np.ones((embeddings_vectors_disease.shape[0], 1)) * 15
    sizes[target_index, :] = 25


    embeddings_vectors_disease = np.concatenate([embeddings_vectors_disease.values[:, 2:], colors, sizes], axis=1)
    embeddings_vectors_disease = pd.DataFrame(embeddings_vectors_disease, columns=['Dim1', 'Dim2', 'color', 'size'], index=all_names)
    embeddings_vectors_disease = embeddings_vectors_disease.astype(float)

    fig2 = go.Figure()  # Create a Figure object directly
    for color in embeddings_vectors_disease['color'].unique():
        filtered_df = embeddings_vectors_disease[embeddings_vectors_disease['color'] == color]
        if filtered_df.shape[0] > 1:
            filtered_df = filtered_df.sample(frac=0.05, random_state=0)
        if filtered_df.shape[0] == 1:
            mycol = 'darkred'
        else:
            mycol = 'salmon'

        fig2.add_trace(go.Scatter(
            x=filtered_df['Dim1'],
            y=filtered_df['Dim2'],
            mode='markers',
            hovertemplate='<b>%{text}</b><br><br>Dim1: %{x}<br>Dim2: %{y}<extra></extra>',
            text=filtered_df.index,  # Add 'name' for hover text
            marker=dict(size=filtered_df['size'].values.squeeze(),
                        color=mycol,
                        line=dict(width=1, color='white')),
        ))

    # --- Layout and Styling ---

    fig2.update_layout(
        xaxis_title="Dim1",
        yaxis_title="Dim2",
        legend_title="Cluster",
        showlegend=False
    )

    fig2.update_layout(coloraxis_showscale=False)
    fig2.update_traces(marker=dict(
        line=dict(width=1,
                  color='white')))
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown("<h6 style='text-align: center; color: grey;'>Only a fraction of the Diseases are displayed to save resources</h6>", unsafe_allow_html=True)



st.write("--------------------------------------------------")
st.markdown("### Link Prediction")
col1, col2 = st.columns(2)
positive_links, negative_links = ml_querier.find_predicted_links_disease(cui=disease_cui)
if positive_links.shape[0] > 0:
    positive_links.columns = ['Microbe', 'Strength', 'CUI']
    positive_links.index = np.array([i + 1 for i in range(len(positive_links))])
if negative_links.shape[0] > 0:
    negative_links.columns = ['Microbe', 'Strength', 'CUI']
    negative_links['Strength'] = - negative_links['Strength']
    negative_links.index = np.array([i + 1 for i in range(len(negative_links))])


with col1:
    st.markdown('### Positive Predicted Links')
    st.table(positive_links)

with col2:
    st.markdown('### Negative Predicted Links')
    st.table(negative_links)