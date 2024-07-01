from langchain_openai import ChatOpenAI
from langchain_community.llms import SagemakerEndpoint
from langchain.llms.sagemaker_endpoint import LLMContentHandler
from typing import Dict, List
import json

from langchain_google_genai import ChatGoogleGenerativeAI
import os
import getpass

if "GOOGLE_API_KEY" in os.environ:
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
elif 'OPENAI_API_KEY':
    OPENAI_MODEL = "gpt-4o"
    llm = ChatOpenAI(
     openai_api_key=os.environ['OPENAI_API_KEY'],
     model=OPENAI_MODEL)

if __name__ == '__main__':
    result = llm.invoke("Write a ballad about LangChain")
    print(result.content)
