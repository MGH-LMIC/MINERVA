from langchain.agents import AgentExecutor, create_react_agent
from langchain import hub
from langchain.tools import Tool
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
import os
from langchain.prompts import PromptTemplate
from llm import llm
from custom_tools import GraphQueries_tool
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper


agent_prompt = PromptTemplate.from_template("""
You are a Microbiologists expert that provides information about microbes and how they are related to diseases.
Be as helpful as possible and return as much information as possible.
Do not answer any questions that do not relate microbes, bacterias or diseases.

Do not answer any questions using your pre-trained knowledge, only use the information provided in the context. 
Usually you will need to use a tool.

TOOLS:
------

You have access to the following tools:

{tools}

To use a tool, please use the following format:

```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```
Thought: Do I need to use a tool? No
Final Answer: [your response here]
```

Begin!

Previous conversation history:
{chat_history}

New input: {input}
{agent_scratchpad}
""")

queries_tool = GraphQueries_tool()


tools = [
    Tool.from_function(
        name="General Chat",
        description="For general chat not covered by other tools",
        func=llm.invoke,
        return_direct=True
        ),

    Tool.from_function(
        name="Evidence inspector",
        description="Use this tool when the user asks to summarize the evidence, to show some references about the relationship between a microbe and a disease, "
                    "or to get some paper IDs (pubmed IDs) or paper titles that talk about a relationship",
        func=queries_tool.query_evidences,
        return_direct=True
        ),

    WikipediaQueryRun(
        name="wiki-tool",
        description="look definitions of the microbes or diseases in wikipedia",
        # args_schema=WikiInputs,
        api_wrapper=WikipediaAPIWrapper(top_k_results=1),
        return_direct=True),


    Tool.from_function(
        name="Evidence inspector of CUIs",
        description="Use this tool when the user asks for the relationship between two CUIs (Concept Unique Identifier) instead of names (they are of the form 'C1006466')",
        func=queries_tool.query_evidences_cuis,
        return_direct=True
        ),


    Tool.from_function(
        name="Strength inspector",
        description="Use this tool when the user specifically about the strength of the relationship between two entities",
        func=queries_tool.query_evidences_strength,
        return_direct=True
        ),

]


memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=5,
    return_messages=True,
)

def generate_response(prompt):
    """
    Create a handler that calls the Conversational agent
    and returns a response to be rendered in the UI
    """

    response = agent_executor.invoke({"input": prompt})

    return response['output']

# Include the LLM from a previous lesson
agent = create_react_agent(llm, tools, agent_prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    handle_parsing_errors=True,
    verbose=True
    )

if __name__ == '__main__':
    generate_response('Summarize all the evidence relating firmicutes and autism in one paragraph')