""" 
[Purpose]
=========
Ingestion of text documents provided within one directory. 
Duplicates are checked and excluded. Imported documents are maintained in an import log.

[References]
============



"""

# [IMPORTS of modules and packages]

import logging
import os # Operating System module for interacting with the OS environment
import uuid # Universally Unique Identifier (UUID) module for generating unique identifiers
import time

from dotenv import load_dotenv # Dotenv library for loading environment variables from a file

import openai # OpenAI library for working with OpenAI's GPT-3 or other models

import pandas as pd # Pandas library for data manipulation and analysis

# Custom module imports related to langchain, a specific library or framework
from langchain.document_loaders import TextLoader  # Text document loading for langchain
from langchain.vectorstores import Chroma  # Vector storage for langchain
from langchain.embeddings import OpenAIEmbeddings  # Embeddings for langchain
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Text splitting for langchain


from chromadb.config import Settings # Importing a specific configuration settings modul
openai.api_key = os.getenv("OPENAI_API_KEY")

# [LOGGING settings]
logging.basicConfig(level = logging.INFO,
                    format = "%(asctime)s : %(levelname)s : Line No. : %(lineno)d - %(message)s",
                    filename = 'braingpt.log',
                    filemode = 'a')
logging.info("Start of script execution")











ROOT_DIR = "/Users/swmoeller/python/2023/large_language_model/BrainGPT"
PROCESSED_DATA_DIR = os.path.join(ROOT_DIR, str(os.getenv("PROCESSED_DATA_DIR")))


# [CONSTANTS definition]
CHROMA_SETTINGS = Settings(
        chroma_db_impl='duckdb+parquet',
        persist_directory=PROCESSED_DATA_DIR,
        anonymized_telemetry=False
)

# Initializing the vector store using OpenAI embeddings.
embeddings = OpenAIEmbeddings()             # type: ignore

in__datastore_location = PROCESSED_DATA_DIR
in__embedding_function = embeddings
in__chromadb_setting = CHROMA_SETTINGS




openai.api_key = os.getenv("OPENAI_API_KEY")

# loading the vectorstore
vectordb = Chroma(
            persist_directory=in__datastore_location,
            embedding_function=in__embedding_function,
            client_settings=in__chromadb_setting
        )