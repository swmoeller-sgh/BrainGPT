#!/usr/bin/env python

""" 
[Purpose]
=========
Ingestion of text documents provided within one directory. 
Duplicates are checked and excluded. Imported documents are maintained in an import log.

[References]
============



"""

# [IMPORTS of modules and packages]

import os # Operating System module for interacting with the OS environment
import uuid # Universally Unique Identifier (UUID) module for generating unique identifiers

from dotenv import load_dotenv # Dotenv library for loading environment variables from a file

import openai # OpenAI library for working with OpenAI's GPT-3 or other models

import pandas as pd # Pandas library for data manipulation and analysis

from typing import List # Typing module for defining type hints

import magic # Magic library for determining the file type of a file

# Custom module imports related to langchain, a specific library or framework
from langchain.document_loaders import TextLoader  # Text document loading for langchain
from langchain.vectorstores import Chroma  # Vector storage for langchain
from langchain.embeddings import OpenAIEmbeddings  # Embeddings for langchain
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Text splitting for langchain
from langchain.llms import OpenAI  # Language Model for langchain
from langchain.chains import VectorDBQA  # Vector Database Question-Answering for langchain
from langchain.chains import RetrievalQA  # Retrieval-based Question-Answering for langchain

from chromadb.config import Settings # Importing a specific configuration settings modul


# [IMPORTS of environment constants]
load_dotenv()   # load environmental variables
root_directory = os.path.dirname(os.path.abspath(__file__))
print("root directory: ", root_directory)

openai.api_key = os.getenv("OPENAI_API_KEY")




# [CONSTANTS definition]
PERSIST_DIRECTORY = ""

CHROMA_SETTINGS = Settings(
        chroma_db_impl='duckdb+parquet',
        persist_directory=PERSIST_DIRECTORY,
        anonymized_telemetry=False
)


  
# [CLASS definition]
  
  
# [FUNCTION definition]

  
# [MODEL definition]

  
# ===================  
# MAIN  
# ===================

if __name__ == "__main__":
    # Code that runs when the script is executed directly
    #execute_main()
    print("Test")
