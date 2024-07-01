import streamlit as st
from graph_queries import GraphQueries

querier = GraphQueries()

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
    layout='wide',
)

# Load CSS from file
with open('styles.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

col1, col2, col3 = st.columns([3, 5, 3])

with col2:
    st.image("icon.png")
    st.write("# Welcome to the MGH-HMS Microbiome-Disease App 👋")

    st.sidebar.success("Select a demo above.")

    st.markdown(
        """
        This is an application for the exploaration of Microbes, Diseases and their relationships in the literature. 
    """
    )

    col1, col2, col3 = st.columns(3)

    st.markdown("""
    <style>
    [data-testid="stMetricLabel"] > div {
        font-size: 30px;  /* Size for the label */
    }
    [data-testid="stMetricValue"] {
        font-size: 70px;  /* Size for the value */
    }
    </style>
    """, unsafe_allow_html=True)

    col1.metric("# Microbes 🦠", querier.count_nodes(label='Microbe'))
    col2.metric("# Diseases 🤢", querier.count_nodes(label='Disease'))
    col3.metric("# Relations 🔗", querier.count_relationships())
