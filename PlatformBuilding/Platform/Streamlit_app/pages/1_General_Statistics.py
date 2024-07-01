import streamlit as st
import pandas as pd
import time
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from graph_queries import GraphQueries

querier = GraphQueries()

st.set_page_config(page_title="General Statistics", page_icon="📈",  layout='wide')

st.markdown("# Statistics about the Database 📈")
st.sidebar.header("Statistics")
st.sidebar.write(
    """General statistics of our database"""
)


st.write("--------------------------------------------------")
# Metrics
col1, col2, col3, col4 = st.columns(4)

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
col3.metric("# M/D Relations 🔗", querier.count_relationships())
col4.metric("# N° Papers with relevant findings 📄", querier.count_papers())
st.write("--------------------------------------------------")


# Rankings of connections
col1, col2 = st.columns(2)
with col1:
    # Ranking of Microbes with more connections
    microbe_connections_rank = querier.get_microbes_with_more_connections_pos_neg(n=10)
    microbe_connections_rank = microbe_connections_rank.iloc[::-1, :]
    microbe_connections_rank.columns = ['Microbe', 'N° of Relations', 'Positive', 'Negative']
    microbe_connections_rank.index = microbe_connections_rank['Microbe']

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=microbe_connections_rank.index,
        x=microbe_connections_rank['Positive'],
        name='Pos',
        orientation='h',
        marker=dict(
            color='rgba(100, 78, 139, 0.6)',
            line=dict(width=2)
        )
    ))
    fig.add_trace(go.Bar(
        y=microbe_connections_rank.index,
        x=microbe_connections_rank['Negative'],
        name='Neg',
        orientation='h',
        marker=dict(
            color='rgba(246, 78, 139, 0.6)',
            line=dict(width=2)
)
    ))
    fig.update_layout(barmode='stack',
                      xaxis_title='N° of Relations',
                      xaxis_title_font_size=20,
                           yaxis_title_font_size=20,
                        title_text="Ranking of Microbes with more Relations <br><span style='font-weight: normal; font-size: 18px;'>More Microbe-Disease relationships</span>",
                        title_font_size=30,
                           # Control tick font sizes
                           xaxis_tickfont_size=18,  # X-axis tick font size
                           yaxis_tickfont_size=18,  # Y-axis tick font size
                            legend_font_size = 20
                      )
    st.plotly_chart(fig, use_container_width=True)
    #fig.write_image("microbes_with_more_connections.svg")



with col2:
    # Ranking of Diseases with more relations
    disease_connections_rank = querier.get_diseases_with_more_connections_pos_neg(n=10)
    disease_connections_rank = disease_connections_rank.iloc[::-1, :]
    disease_connections_rank.columns = ['Disease', 'N° of Relations', 'Positive', 'Negative']
    disease_connections_rank.index = disease_connections_rank['Disease']

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=disease_connections_rank.index,
        x=disease_connections_rank['Positive'],
        name='Pos',
        orientation='h',
        marker=dict(
            line=dict(width=2)
        )
    ))
    fig.add_trace(go.Bar(
        y=disease_connections_rank.index,
        x=disease_connections_rank['Negative'],
        name='Neg',
        orientation='h',
        marker=dict(
            #color='rgba(246, 78, 139, 0.6)',
            line=dict(width=2)
)
    ))
    fig.update_layout(barmode='stack',
                      xaxis_title='N° of Relations',
                      xaxis_title_font_size=20,
                           yaxis_title_font_size=20,
                      title_text="Ranking of Diseases with more Relations <br><span style='font-weight: normal; font-size: 18px;'>More Microbe-Disease relationships</span>",
                      title_font_size=30,
                           # Control tick font sizes
                           xaxis_tickfont_size=18,  # X-axis tick font size
                           yaxis_tickfont_size=18,  # Y-axis tick font size
                            legend_font_size = 20
                      )

    #fig.write_image("diseases_with_more_connections.svg")
    st.plotly_chart(fig, use_container_width=True)

st.write("--------------------------------------------------")




# Rankings
col1, col2 = st.columns(2)
with col1:
    # Ranking of Microbes with more references
    microbe_references_rank = querier.get_microbes_with_more_references_pos_neg(n=10)
    microbe_references_rank = microbe_references_rank.iloc[::-1, :]
    microbe_references_rank.columns = ['Microbe', 'N° of Relations', 'Positive', 'Negative']
    microbe_references_rank.index = microbe_references_rank['Microbe']

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=microbe_references_rank.index,
        x=microbe_references_rank['Positive'],
        name='Pos',
        orientation='h',
        marker=dict(
            color='#ACE894',
            line=dict(width=2)
        )
    ))
    fig.add_trace(go.Bar(
        y=microbe_references_rank.index,
        x=microbe_references_rank['Negative'],
        name='Neg',
        orientation='h',
        marker=dict(
            color='#A8BA9A',
            line=dict(width=2)
)
    ))
    fig.update_layout(barmode='stack',
                      xaxis_title_font_size=20,
                      xaxis_title='N° of References',
                           yaxis_title_font_size=20,
                        title_text="Ranking of Microbes with more Mentions  <br><span style='font-weight: normal; font-size: 18px;'>Microbe-Disease Mentions in the Literature</span>",
                        title_font_size=30,
                           # Control tick font sizes
                           xaxis_tickfont_size=18,  # X-axis tick font size
                           yaxis_tickfont_size=18,  # Y-axis tick font size
                            legend_font_size = 20
                      )
    #fig.write_image("fig1.pdf")

    st.plotly_chart(fig, use_container_width=True)



with col2:
    # Ranking of Diseases with more References
    disease_references_rank = querier.get_diseases_with_more_references_pos_neg(n=10)
    disease_references_rank = disease_references_rank.iloc[::-1, :]
    disease_references_rank.columns = ['Disease', 'N° of Relations', 'Positive', 'Negative']
    disease_references_rank.index = disease_references_rank['Disease']

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=disease_references_rank.index,
        x=disease_references_rank['Positive'],
        name='Pos',
        orientation='h',
        marker=dict(
            line=dict(width=2),
            color='#C6D8D3',
        )
    ))
    fig.add_trace(go.Bar(
        y=disease_references_rank.index,
        x=disease_references_rank['Negative'],
        name='Neg',
        orientation='h',
        marker=dict(
            color='#006DC6',
            line=dict(width=2)
)
    ))
    fig.update_layout(barmode='stack',
                      xaxis_title_font_size=20,
                           yaxis_title_font_size=20,
                      xaxis_title='N° of References',
                      title_text="Ranking of Diseases with more Mentions  <br><span style='font-weight: normal; font-size: 18px;'>Microbe-Disease Mentions in the Literature</span>",
                      title_font_size=30,
                           # Control tick font sizes
                           xaxis_tickfont_size=18,  # X-axis tick font size
                           yaxis_tickfont_size=18,  # Y-axis tick font size
                            legend_font_size = 20
                      )
    #fig.write_image("fig2.pdf")

    st.plotly_chart(fig, use_container_width=True)

st.write("--------------------------------------------------")


# Journals
col1, col2 = st.columns(2)

with col1:
    publications_by_journal = querier.get_publications_by_journal(n=10)
    publications_by_journal = publications_by_journal.iloc[::-1, :]
    publications_by_journal.columns = ['N° of Publications']
    publications_by_journal.index.rename('Journal', inplace=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=publications_by_journal.index,
        x=publications_by_journal['N° of Publications'],
        orientation='h',
        marker=dict(
            color='rgba(100, 78, 250, 0.6)',
            line=dict(width=2)
        )
    ))
    fig.update_layout(xaxis_title='N° of publications',
                      xaxis_title_font_size=20,
                      yaxis_title_font_size=20,
                      title_text="Publications by Journal <br><span style='font-weight: normal; font-size: 18px;'>Journals with more relevant publications</span>",
                      title_font_size=30,
                      # Control tick font sizes
                      xaxis_tickfont_size=18,  # X-axis tick font size
                      yaxis_tickfont_size=18,  # Y-axis tick font size
                      legend_font_size=20
                      )

    st.plotly_chart(fig, use_container_width=True)
    #fig.write_image("journals.svg")


with col2:
    publications_by_year = querier.get_publications_by_year()
    publications_by_year.columns = ['Year', 'Publications']

    relationships_by_year = querier.get_relationships_by_year()
    relationships_by_year.columns = ['Year', 'Relationships']

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=publications_by_year['Year'], y=publications_by_year['Publications'], fill='tozeroy', name='Pub.', fillcolor='rgba(246, 78, 139, 0.6)', line_color='rgba(246, 78, 139, 0.6)'),  secondary_y=False)
    fig.add_trace(go.Scatter(x=publications_by_year['Year'], y=relationships_by_year['Relationships'], fill='tozeroy', name='Rel.'), secondary_y=True)

    fig.update_layout(
        xaxis_title="Year",
        yaxis_title="N° Publications",
        yaxis2_title="N° Relationships",
        xaxis_anchor="y1",  # Ensure both charts share the same x-axis

        xaxis_title_font_size=20,
        yaxis_title_font_size=20,
        yaxis2_title_font_size=20,
        title_text="Publications-Relationships by year <br><span style='font-weight: normal; font-size: 18px;'>How many publications by year and how many new relationships were found</span>",
        title_font_size=30,
        # Control tick font sizes
        xaxis_tickfont_size=18,  # X-axis tick font size
        yaxis_tickfont_size=18,  # Y-axis tick font size
        yaxis2_tickfont_size=18,  # Y-axis tick font size
        legend_font_size=20
    )
    #fig.write_image("general_publications.svg")
    st.plotly_chart(fig, use_container_width=True)

st.write("--------------------------------------------------")
# Ranking of strength
col1, col2 = st.columns(2)

with col1:
    st.markdown("## Strongest positive relations")
    st.markdown("""
    <style>
    table {
        font-family: 'Arial', sans-serif; 
        border-collapse: collapse;
        font-size:18px;

    }
    th {
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)
    positive_microbe_disease = querier.rank_by_positive_strength(n=10)
    positive_microbe_disease['Ranking'] = np.array([i + 1 for i in range(10)])
    positive_microbe_disease.index = positive_microbe_disease['Ranking']
    del positive_microbe_disease['Ranking']
    st.table(positive_microbe_disease)

with col2:
    st.markdown("## Strongest negative relations")
    st.markdown("""
    <style>
    table {
        font-family: 'Arial', sans-serif; 
        border-collapse: collapse;
        font-size:18px;
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
    negative_microbe_disease = querier.rank_by_negative_strength(n=10)
    negative_microbe_disease['Ranking'] = np.array([i + 1 for i in range(10)])
    negative_microbe_disease.index = negative_microbe_disease['Ranking']
    del negative_microbe_disease['Ranking']
    st.table(negative_microbe_disease)