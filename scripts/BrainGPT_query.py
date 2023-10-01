"""
[Purpose]
=========
Query our own data

[References]
============

"""

# TODO: Consider moving to local processing for better efficiency. Check the LinkedIn article and Obsidian notes for details.
# TODO: Implement an empty form at the beginning for user input.
# FIXME: Address the issue of an empty historic chat at the beginning.

# [IMPORTS of modules and packages]

import logging
import os  # Operating System module for interacting with the OS environment

from dotenv import load_dotenv, find_dotenv # Dotenv library for loading env. variables from a file

import openai  # OpenAI library for working with OpenAI's GPT-3 or other models

#import pandas as pd  # Pandas library for data manipulation and analysis

# Custom module imports related to langchain, a specific library or framework
# from langchain.vectorstores import Chroma  # Vector storage for langchain
# from langchain.embeddings import OpenAIEmbeddings  # Embeddings for langchain
# from langchain.text_splitter import RecursiveCharacterTextSplitter  # Text splitting for langchain

from flask import Flask

# from braingpt_src.routes import *

# from chromadb.config import Settings # Importing a specific configuration settings module

# [IMPORTS of environment constants]

# [INITIALIZE environment]
env_file = find_dotenv(".env")
load_dotenv(env_file)

# Set the OpenAI API key from environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")

# Define the root directory for file operations
ROOT_DIR = os.path.join(str(os.getenv("ROOT_DIR")))

# Define the path to the query log file
BRAINGPT_INQUIRY_FILE = os.path.join(ROOT_DIR, str(os.getenv("LOG_DIR")),
                                     str(os.getenv("QUERY_LOG_NAME")))

# [LOGGING settings]

# Configure logging settings
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s : %(levelname)s : Line No. : %(lineno)d - %(message)s",
                    filename=BRAINGPT_INQUIRY_FILE,
                    filemode='a')
logging.info("Start of script execution: %s", os.path.basename(__file__))

# Define if local testing is done (without inquiring OpenAI)
DEBUG_MODE = os.getenv("DEBUG", "False") == "True"
logging.info("DEBUG mode: %s", DEBUG_MODE)

# Definition of GPT-relevant inputs
app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY")

# main part

if __name__ == "__main__":
    # Start the Flask application on host 127.0.0.1 and port 4455 for debugging
    app.run(host="127.0.0.1",  # type: ignore
            port=4455,
            debug=True)

    