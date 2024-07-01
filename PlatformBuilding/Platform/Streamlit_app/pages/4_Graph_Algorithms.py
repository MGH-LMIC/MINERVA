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

st.set_page_config(page_title="Graph Analysis", page_icon="🕸️",  layout='wide')

st.markdown("# Graph Analysis 🕸️")
st.sidebar.header("Explore our Graph!")
st.sidebar.write(
    """Here you will be able to analyze the families of Microbes and Diseases in our Graph"""
)

st.write("--------------------------------------------------")



# Path Finding
st.markdown("## Path Finding")
st.markdown("##### We use Dijkstra algorithm to find the shortest path between a source node and a target node")
col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Select a Source node")
    source_type = st.selectbox(label='Select the type of the Source Node', options=['Microbe', 'Disease'],
                               index=0, key=15)
    if source_type == 'Microbe':
        source = querier.get_all_microbes()
    else:
        source = querier.get_all_diseases()

    names = [elem.capitalize() for elem in source['name'].values.squeeze()]
    cuis = [elem for elem in source['cui'].values.squeeze()]
    names = ['{} ({})'.format(names[i], cuis[i]) for i in range(len(cuis))]
    option = st.selectbox(label='hola', options=names, index=0, label_visibility='hidden', key=10)
    source_name = option.split('(C')[0].strip()
    source_cui = option.split('(C')[1]
    source_cui = 'C' + source_cui[:-1]

with col2:
    st.markdown("#### Select a Target Node")
    target_type = st.selectbox(label='Select the type of the Target Node', options=['Disease', 'Microbe'],
                               index=0, key=25)
    if target_type == 'Microbe':
        target = querier.get_all_microbes()
    else:
        target = querier.get_all_diseases()

    names = [elem.capitalize() for elem in target['name'].values.squeeze()]
    cuis = [elem for elem in target['cui'].values.squeeze()]
    names = ['{} ({})'.format(names[i], cuis[i]) for i in range(len(cuis))]
    option = st.selectbox(label='hola', options=names, index=0, label_visibility='hidden', key=11)
    target_name = option.split('(C')[0].strip()
    target_cui = option.split('(C')[1]
    target_cui = 'C' + target_cui[:-1]

# Shortest path
source_node = {'label': source_type, 'cui': source_cui}
target_node = {'label': target_type, 'cui': target_cui}
shortest_path = ml_querier.shortest_path(source_node, target_node)

# Draw Graph
# Get all nodes
sources = shortest_path['Source'].values
source_types = shortest_path['SourceType'].values
targets = shortest_path['Target'].values
target_types = shortest_path['TargetType'].values

nodes = [(sources[i], source_types[i]) for i in range(len(sources))] + [(targets[i], target_types[i]) for i in range(len(sources))]
nodes = list(set(nodes))
nodes = [Node(id=elem[0], label=elem[0], size=25, color='#F79767', font={'color': '#474747', 'size':'16'}) if elem[1] == 'Microbe' else
         Node(id=elem[0], label=elem[0], size=25, color='#C990C0', font={'color': '#474747', 'size':'16'})for elem in nodes]

edges = []
shortest_path = shortest_path.drop_duplicates()

for kk in range(len(shortest_path)):
    row = shortest_path.iloc[kk]
    if row['Relation'] == 'STRENGTH':
        edges.append(Edge(source=row['Source'],
                      label=str(row['Strength']),
                      font={'color': '#9e9e9e', 'size': '15', 'strokeWidth': '0'},
                      target=row['Target']))
    else:
        edges.append(Edge(source=row['Source'],
                          label=row['Relation'],
                          font={'color': '#9e9e9e', 'size': '15', 'strokeWidth': '0'},
                          target=row['Target']))

config = Config(
    height='400',
    width='1500',
    directed=True,
    physics=True,
    hierarchical=False,
    highlightColor="#F7A7A6",
    maxZoom=2,
    minZoom=0.1,
    # **kwargs
)

return_value = agraph(nodes=nodes,
                      edges=edges,
                      config=config)

st.write("--------------------------------------------------")




# Graph connections
