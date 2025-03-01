import streamlit as st
import pandas as pd
from streamlit_agraph import agraph, Node, Edge, Config
from graph_queries import GraphQueries
import numpy as np
import plotly.graph_objects as go
import copy


querier = GraphQueries()

st.set_page_config(page_title="Microbiome Analysis", page_icon="🦠",  layout='wide')

st.markdown("# Microbiome Analysis 🦠")
st.sidebar.header("Specific Analysis")
st.sidebar.write(
    """Specific analysis of selected Microbes"""
)

st.write("--------------------------------------------------")


microbes = querier.get_all_microbes()
names = [elem.capitalize() for elem in microbes['name'].values.squeeze()]
cuis = [elem for elem in microbes['cui'].values.squeeze()]
names = ['{} ({})'.format(names[i], cuis[i]) for i in range(len(cuis))]

col1, col2, col3 = st.columns([1, 4, 1])

with col2:
    st.markdown("<h2 style='text-align: center;'>Select a Microbe to start</h2>", unsafe_allow_html=True)
    options = st.multiselect(label='hola', options=names, default=names[0], label_visibility='hidden',
                             max_selections=4)

    # Extracting the CUI
    options = [elem.split('(C')[1] for elem in options]
    options = ['C' + elem[:-1] for elem in options]

st.write("--------------------------------------------------")

st.markdown('## Basic Information')
# Fiding all about the microbe
microbe_infos = [querier.get_microbe_by_property({'cui': option}) for option in options]


st.markdown("""
<style>
.container {
    width: 70%; /* Adjust width as needed */
    font-size:20px;
}

.container_def {
    width: 100%; /* Adjust width as needed */
    font-size:16px;
}

.rectangle {
    background-color: #ECECEC;
    padding: 10px;
    border-radius: 10px; /* Rounded corners */
}


</style>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

# Example usage
with col1:
    official_name_text = ["{}) {}".format(i +1, microbe_infos[i]['official_name']) for i in range(len(microbe_infos))]
    official_name_text = '&#10;'.join(official_name_text)
    st.markdown('### Official Name')
    st.markdown('<div class="container rectangle" style="white-space: pre-wrap;"> {} </div>'.format(official_name_text), unsafe_allow_html=True)

    st.markdown('### Rank')
    rank = [microbe_infos[i]['rank'].capitalize() if type(microbe_infos[i]['rank']) == str else microbe_infos[i]['rank']
            for i in range(len((microbe_infos)))]
    rank = ["{}) {}".format(i + 1, rank[i]) for i in range(len(rank))]
    rank = '&#10;'.join(rank)

    st.markdown('<div class="container rectangle" style="white-space: pre-wrap;"> {} </div>'.format(rank), unsafe_allow_html=True)

with col2:
    st.markdown('### CUI')
    cui_text = ["{}) {}".format(i + 1, microbe_infos[i]['cui']) for i in range(len(microbe_infos))]
    cui_text = '&#10;'.join(cui_text)
    st.markdown('<div class="container rectangle" style="white-space: pre-wrap;"> {} </div>'.format(cui_text), unsafe_allow_html=True)

    st.markdown('### Tax ID')
    taxid_text = ["{}) {}".format(i + 1, microbe_infos[i]['tax_id']) for i in range(len(microbe_infos))]
    taxid_text = '&#10;'.join(taxid_text)
    st.markdown('<div class="container rectangle" style="white-space: pre-wrap;"> {} </div>'.format(taxid_text), unsafe_allow_html=True)


with col3:
    st.markdown('### Definition')
    definition_text = ["{}) {}".format(i + 1, microbe_infos[i]['definition']) for i in range(len(microbe_infos))]
    definition_text = '&#10;'.join(definition_text)
    st.markdown('<div class="container_def rectangle" style="white-space: pre-wrap;"> {} </div>'.format(definition_text), unsafe_allow_html=True)

st.write("--------------------------------------------------")



# Define desired border styles
col1, col2 = st.columns(2)

with col1:
    col_11, col12 = st.columns(2)
    with col_11:
        st.markdown("<h3 style='text-align: center;'>Pos. Relations </h3>", unsafe_allow_html=True)
        positive_relations = pd.concat([querier.get_microbe_relations(cui=option, rel_type='POSITIVE') for option in options], axis=0)
        if positive_relations.shape[0] > 0:
            positive_relations.columns = ['Microbe Name', 'Disease Name', 'Strength', 'Disease CUI']
            positive_relations.index = positive_relations['Microbe Name']
            del positive_relations['Microbe Name']
        st.dataframe(positive_relations, use_container_width=True)

    with col12:
        st.markdown("<h3 style='text-align: center;'>Neg. Relations </h3>", unsafe_allow_html=True)
        negative_relations = pd.concat([querier.get_microbe_relations(cui=option, rel_type='NEGATIVE') for option in options], axis=0)
        if negative_relations.shape[0] > 0:
            negative_relations.columns =['Microbe Name', 'Disease Name', 'Strength', 'Disease CUI']
            negative_relations.index = negative_relations['Microbe Name']
            del negative_relations['Microbe Name']
        st.dataframe(negative_relations, use_container_width=True)

with col2:
    st.markdown("<h3 style='text-align: center;'>Graph</h3>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: center;'>(At most only 50 relations per node are shown)</h5>", unsafe_allow_html=True)

    graph = pd.concat([querier.find_one_hop_microbe(cui=option) for option in options], axis=0).astype(str)

    # Get all nodes
    sources = graph['source'].values
    source_types = graph['source_type'].values
    targets = graph['target'].values
    target_types = graph['target_type'].values

    nodes = [(sources[i], source_types[i]) for i in range(len(sources))]  + [(targets[i], target_types[i]) for i in range(len(sources))]
    nodes = list(set(nodes))
    nodes = [Node(id=elem[0], label=elem[0], size=25, color='#F79767', font={'color': '#474747', 'size':'16'}) if elem[1] == 'Microbe' else
             Node(id=elem[0], label=elem[0], size=25, color='#C990C0', font={'color': '#474747', 'size':'16'})for elem in nodes]

    edges = []

    graph = graph.drop_duplicates()

    for kk in range(len(graph)):
        row = graph.iloc[kk]
        if row['relation'] == 'STRENGTH':
            continue
        edges.append(Edge(source=row['source'],
                          label=row['relation'],
                          font={'color': '#9e9e9e', 'size': '15', 'strokeWidth':'0'},
                          target=row['target']))

    config = Config(
        height='400',
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


# Publications of the microbe by year
st.write("--------------------------------------------------")
fig = go.Figure()
publications_by_year_all = [querier.popularity_in_time(label='Microbe', cui=option) for option in options]
indices = np.concatenate([elem['publication_year'].astype(int) for elem in publications_by_year_all], axis=0)
new_index = [i for i in range(indices.min(), indices.max() + 1)]
for i in range(len(options)):
    publications_by_year = querier.popularity_in_time(label='Microbe', cui=options[i])
    publications_by_year.columns = ['Year', 'N° of Publications']
    publications_by_year.index = publications_by_year['Year'].astype(int)
    del publications_by_year['Year']
    publications_by_year = publications_by_year.reindex(new_index)
    publications_by_year = publications_by_year.fillna(0)

    fig.add_trace(go.Scatter(x=publications_by_year.index, y=publications_by_year['N° of Publications'], name='Pub. {}'.format(microbe_infos[i]['name']),
                             fill='tozeroy', marker=dict(size=12)))
fig.update_xaxes(type='category')
fig.update_layout(
        xaxis_title="Year",
        yaxis_title="N° of Publications",
        xaxis_title_font_size=20,
        yaxis_title_font_size=20,
        title_text="Popularity of selected microbes by year <br><span style='font-weight: normal; font-size: 18px;'> "
                   "Number of publications related to selected microbes in our database </span>",
        title_font_size=30,
        # Control tick font sizes
        xaxis_tickfont_size=18,  # X-axis tick font size
        yaxis_tickfont_size=18,  # Y-axis tick font size
        legend_font_size=20
    )
fig.write_image("publications_of_microbe.svg")

st.plotly_chart(fig, use_container_width=True)





st.write("--------------------------------------------------")
st.markdown("<h3>Publications Detail</h3>", unsafe_allow_html=True)
related_publications = pd.concat([querier.get_related_publications_microbe(cui=option) for option in options], axis=0)
if related_publications.shape[0] > 0:
    related_publications.columns = ['PMID', 'Microbe', 'Related Disease', 'Rel Type', 'Year', 'Journal', 'Title', 'Evidence']
    related_publications.index = related_publications['PMID']
    del related_publications['PMID']

st.markdown("""
<style>
    .dataframe div[data-testid="row"] {  /* Targets table cells */
        font-size: 20px;
    }
    .dataframe thead th {  /* Targets header */
        font-size: 20px;
    }
</style>
""", unsafe_allow_html=True)
st.dataframe(related_publications, use_container_width=True)




