import py2neo
import streamlit as st
from py2neo import Graph, NodeMatcher
from dataclasses import dataclass
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from transformers import AutoTokenizer, AutoModel
from llm import llm
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.agents import AgentExecutor, create_react_agent


class entitiesParser(BaseModel):
    microbe: str = Field(description="microbe or taxonomic species mentioned in the prompt")
    disease: str = Field(description="disease or human heath condition mentioned in the prompt")

# Set up a parser + inject instructions into the prompt template.
parser = JsonOutputParser(pydantic_object=entitiesParser)

class MyEmbeddingFunction(EmbeddingFunction):

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext")
        self.similarity_model = AutoModel.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext").cuda()

    def __call__(self, input) -> Embeddings:
        # embed the documents somehow
        toks = self.tokenizer.batch_encode_plus(input,
                                           padding="max_length",
                                           max_length=25,
                                           truncation=True,
                                           return_tensors="pt")
        toks_cuda = {}
        for k, v in toks.items():
            toks_cuda[k] = v.cuda()
        embeddings = self.similarity_model(**toks_cuda)[0][:, 0, :].detach().cpu().numpy()  # use CLS representation as the embedding
        return [elem.tolist() for elem in embeddings]

@dataclass(frozen=True)
class GraphQueries_tool:
    graph = Graph(host="0.0.0.0", port="7687", auth=('neo4j', 'your_password'))
    chroma_client = chromadb.PersistentClient(path="embeddings/chormadb")

    chroma_microbes_collection = chroma_client.get_collection('microbes_names_embedding', embedding_function=MyEmbeddingFunction())
    chroma_diseases_collection = chroma_client.get_collection('diseases_names_embedding', embedding_function=MyEmbeddingFunction())

    def get_cuis(self, microbe, disease):
        similar_microbes = self.chroma_microbes_collection.query(query_texts=[microbe], n_results=1)['metadatas'][0][0]
        similar_diseases = self.chroma_diseases_collection.query(query_texts=[disease], n_results=1)['metadatas'][0][0]

        return similar_microbes, similar_diseases

    def entity_extraction(self, prompt):
        final_prompt = (
            "Given the following prompt, extract the mentioned microbe and disease. \n{format_instructions} \n Here is the prompt: {prompt}")
        prompt_template = PromptTemplate(
            template=final_prompt,
            input_variables=["prompt"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        chain = prompt_template | llm | parser
        out = chain.invoke({'prompt': prompt})
        return out

    def cuis_extraction(self, prompt):
        final_prompt = ("Given the following prompt, extract all the mentioned CUIs separated by comma (they are of the form 'C1006466') "
                        "\n Here is the prompt {}".format(prompt))
        out = llm.invoke(final_prompt)
        out = out.content.strip().split(',')

        return out

    @st.cache_data
    def query_evidences(self, prompt):
        try:
            original_entities = self.entity_extraction(prompt)
            print('Extracted entities from prompt: {}'.format(original_entities))
        except:
            return "Can you specify in your prompt which microbe and which disease are you talking about?"

        microbe_dict, disease_dict = self.get_cuis(original_entities['microbe'].lower(), original_entities['disease'].lower())
        print('Closest microbe: {}'.format(microbe_dict))
        print('Closest disease: {}'.format(disease_dict))

        query = """
        Match (m:Microbe)-[r:POSITIVE|NEGATIVE]-(d:Disease)
        Where m.cui='{}' and d.cui='{}'
        Return r.pmid as PaperID, r.evidence as Evidence, type(r) as RelDirection, r.title as PaperTitle
        """.format(microbe_dict['cui'], disease_dict['cui'])

        result = self.graph.run(query).to_data_frame()
        result = result.to_string()


        prompt = prompt + ('\n All the information that you need is in the following dataframe, the column PaperID are '
                           'paper ids (IDs from pubmed), the column Evidence are relevant sentences from the paper, the '
                           'column RelDirection is the direction of the relationship between microbe {} ({}) and disease {} ({}) and the column PaperTitle'
                           'is the title of each paper. \n '
                           'Here is the dataframe: {}'
                           .format(original_entities['microbe'], microbe_dict['name'], original_entities['disease'],
                                   disease_dict['name'], result))

        out = llm.invoke(prompt)
        return out.content

    @st.cache_data
    def query_evidences_cuis(self, prompt):

        try:
            original_entities = self.cuis_extraction(prompt)
            print('Extracted entities from prompt: {}'.format(original_entities))
        except:
            return "Can you specify in your prompt which CUIs are you talking about?"

        # Getting the names of the entities
        query = """
        Match (m)
        WHERE m.cui = '{}'
        Return m.name as name, m.official_name as official_name
        """
        answer1 = self.graph.run(query.format(original_entities[0].strip())).data()[0]
        name1 = answer1['name']
        official_name1 = answer1['official_name']

        answer2 = self.graph.run(query.format(original_entities[1].strip())).data()[0]
        name2 = answer2['name']
        official_name2 = answer2['official_name']
        print('Extracted names 1: {} | {}'.format(name1, official_name1))
        print('Extracted names 2: {} | {}'.format(name2, official_name2))

        query = """
           Match (m)-[r:POSITIVE|NEGATIVE]-(d)
           Where m.cui='{}' and d.cui='{}'
           Return r.pmid as PaperID, r.evidence as Evidence, type(r) as RelDirection, r.title as PaperTitle
           """.format(original_entities[0].strip(), original_entities[1].strip())

        result = self.graph.run(query).to_data_frame()
        result = result.to_string()

        prompt = prompt + ('\n All the information that you need is in the following dataframe, the column PaperID are '
                           'paper ids (IDs from pubmed), the column Evidence are relevant sentences from the paper, the '
                           'column RelDirection is the direction of the relationship between {} (which is equivalent to {} or {}) '
                           'and {} (which is equivalent to {} or {}). Finally, the column PaperTitle'
                           'is the title of each paper. \n  '
                           'Here is the dataframe: {}'
                           .format(original_entities[0].strip(), name1, official_name1, original_entities[0].strip(),
                                   name2, official_name2, result))

        out = llm.invoke(prompt)
        return out.content

    @st.cache_data
    def query_evidences_strength(self, prompt):
        try:
            original_entities = self.entity_extraction(prompt)
            print('Extracted entities from prompt: {}'.format(original_entities))
        except:
            return "Can you specify in your prompt which microbe and which disease are you talking about?"

        microbe_dict, disease_dict = self.get_cuis(original_entities['microbe'].lower(),
                                                   original_entities['disease'].lower())
        print('Closest microbe: {}'.format(microbe_dict))
        print('Closest disease: {}'.format(disease_dict))

        query = """
            Match (m:Microbe)-[r:STRENGTH]-(d:Disease)
            Where m.cui='{}' and d.cui='{}'
            Return r.strength_IF as Strength
            """.format(microbe_dict['cui'], disease_dict['cui'])

        result = self.graph.run(query).to_data_frame()
        result = result.to_string()

        prompt = prompt + ('\n All the information that you need is in the following dataframe. Here the Strength column shows the '
                           'strength of the relationship between microbe {} ({}) and disease {} ({}). \n '
                           'Consider that a Positive Strength means that the microbe and the disease are positively correlated and a '
                           'negative Strength means that the microbe and the disease are negatively correlated'
                           'Here is the dataframe: {}'
                           .format(original_entities['microbe'], microbe_dict['name'], original_entities['disease'],
                                   disease_dict['name'], result))

        out = llm.invoke(prompt)
        return out.content







if __name__ == '__main__':
    # Give me five references that talk about the relationship between firmicutes and autism and summarize each of them in one sentence

    queries_tool = GraphQueries_tool()
    #queries_tool.query_evidences(prompt="Summarize all the evidence relating firmicutes and autism in one paragraph")
    queries_tool.query_evidences_cuis(prompt="Summarize all the evidence relating 'C1254144' and 'C0004352' in one paragraph")
    # df = queries_tool.query_evidences(microbe='bacillota', disease='autism')
    #
    # agent = create_pandas_dataframe_agent(
    #     llm,
    #     df,
    #     verbose=True,
    #     agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    #     allow_dangerous_code=True,
    # )
    #
    # out = agent.invoke("Give me specific references that say that the relation between firmicutes and autism is "
    #                    "positive and summarize what each of the selected references say in no more than one sentence",
    #                    handle_parsing_errors=True)
