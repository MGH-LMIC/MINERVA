import os
import streamlit as st
from utils import write_message
import os

st.set_page_config(page_title="Chat with our database", page_icon="👦🏻🤖",  layout='wide')

st.markdown("# Chat with our DB 👦🏻🤖")
st.sidebar.header("Chat with the Database")
st.sidebar.write(
    """Here you can do more specific queries in Natural Language to chat with our Database"""
)

openai_api_key = st.sidebar.text_input("OpenAI API Key", type="password")
gemini_api_key = st.sidebar.text_input("Gemini API Key", type="password")

if len(gemini_api_key) > 1:
    os.environ['GOOGLE_API_KEY'] = gemini_api_key
elif len(openai_api_key) > 1:
    os.environ['OPENAI_API_KEY'] = openai_api_key

from agent import generate_response

st.write("--------------------------------------------------")


# tag::session[]
# Set up Session State
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "assistant", "content": "Hi, I'm the Microbiome MGH-HMS Chatbot!  How can I help you?"},
    ]
# end::session[]

# tag::submit[]
# Submit handler
def handle_submit(message):
    """
    Submit handler:

    You will modify this method to talk with an LLM and provide
    context using data from Neo4j.
    """

    # Handle the response
    with st.spinner('Thinking...'):
        # # TODO: Replace this with a call to your LLM
        #from time import sleep
        #sleep(1)
        response = generate_response(message)
        write_message('assistant', response)
# end::submit[]


# tag::chat[]
# Display messages in Session State
for message in st.session_state.messages:
    write_message(message['role'], message['content'], save=False)

# Handle any user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    write_message('user', prompt)

    # Generate a response
    handle_submit(prompt)
# end::chat[]