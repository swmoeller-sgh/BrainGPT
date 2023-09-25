#!/usr/bin/env python

""" 
[Purpose]
=========
Ingestion of text documents provided within one directory. 
Duplicates are checked and excluded. Imported documents are maintained in an import log.

[References]
============



"""
ROOT_DIR = "/Users/swmoeller/python/2023/large_language_model/BrainGPT"
# // TODO - using function from utilities_braingpt, automatize identification of root-directory


# [IMPORTS of modules and packages]

import logging 

from dotenv import load_dotenv # Dotenv library for loading environment variables from a file

import os # Operating System module for interacting with the OS environment

import uuid # Universally Unique Identifier (UUID) module for generating unique identifiers
from typing import List # Typing module for defining type hints
import magic # Magic library for determining the file type of a file

import openai # OpenAI library for working with OpenAI's GPT-3 or other models

import pandas as pd # Pandas library for data manipulation and analysis

# Custom module imports related to langchain, a specific library or framework
from langchain.document_loaders import TextLoader  # Text document loading for langchain
from langchain.vectorstores import Chroma  # Vector storage for langchain
from langchain.embeddings import OpenAIEmbeddings  # Embeddings for langchain
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Text splitting for langchain
from langchain.llms import OpenAI  # Language Model for langchain
from langchain.chains import VectorDBQA  # Vector Database Question-Answering for langchain
from langchain.chains import RetrievalQA  # Retrieval-based Question-Answering for langchain


from langchain.document_loaders import (
    CSVLoader,  # Handles loading text from CSV files.
    EverNoteLoader,  # Handles loading text from Evernote documents.
    PDFMinerLoader,  # Handles loading text from PDF documents using PDFMiner.
    TextLoader,  # Handles loading plain text documents.
    UnstructuredEmailLoader,  # Handles loading text from unstructured email messages.
    UnstructuredEPubLoader,  # Handles loading text from unstructured EPUB documents.
    UnstructuredHTMLLoader,  # Handles loading text from unstructured HTML documents.
    UnstructuredMarkdownLoader,  # Handles loading text from unstructured Markdown files.
    UnstructuredODTLoader,  # Handles loading text from unstructured ODT (OpenDocument Text) files.
    UnstructuredPowerPointLoader,  # Handles loading text from unstructured PowerPoint documents.
    UnstructuredWordDocumentLoader,  # Handles loading text from unstructured Word documents.
)

from chromadb.config import Settings # Importing a specific configuration settings modul


# [LOGGING settings]
logging.basicConfig(level = logging.INFO,
                    format = "Date-Time : %(asctime)s : %(levelname)s : Line No. : %(lineno)d - %(message)s",
                    filename = 'braingpt.log',
                    filemode = 'a')
logging.info("Start of script execution")


# [IMPORTS of environment constants]
load_dotenv()   # load environmental variables

openai.api_key = os.getenv("OPENAI_API_KEY")

PROCESSED_DATA_DIR = os.path.join(ROOT_DIR, os.getenv("PROCESSED_DATA_DIR"))
logging.info("Storage for processed data: %s",PROCESSED_DATA_DIR)

IMPORT_LOG_FILE = os.path.join(ROOT_DIR, os.getenv("LOG_DIR"), os.getenv("IMPORT_LOG_NAME"))
logging.info("Log file on imported documents: %s",IMPORT_LOG_FILE)

DOC2SCAN_DATA_DIR = os.path.join(ROOT_DIR, os.getenv("DOC2SCAN_DATA_DIR"))
logging.info("Source directory for documents: %s",DOC2SCAN_DATA_DIR)


# [CONSTANTS definition]
CHROMA_SETTINGS = Settings(
        chroma_db_impl='duckdb+parquet',
        persist_directory=PROCESSED_DATA_DIR,
        anonymized_telemetry=False
)

LOADER_MAPPING = {
    ".csv": (CSVLoader, {}),
    # ".docx": (Docx2txtLoader, {}),
    ".doc": (UnstructuredWordDocumentLoader, {}),
 #   ".docx": (UnstructuredWordDocumentLoader, {}),
    ".enex": (EverNoteLoader, {}),
 #   ".eml": (MyElmLoader, {}),
    ".epub": (UnstructuredEPubLoader, {}),
    ".html": (UnstructuredHTMLLoader, {}),
    ".md": (UnstructuredMarkdownLoader, {}),
    ".odt": (UnstructuredODTLoader, {}),
    ".pdf": (PDFMinerLoader, {}),
    ".ppt": (UnstructuredPowerPointLoader, {}),
    ".pptx": (UnstructuredPowerPointLoader, {}),
    ".txt": (TextLoader, {"encoding": "utf8"}),
    # Add more mappings for other file extensions and loaders as needed
}





# [CLASS definition]


# [MODEL definition]


# [FUNCTION definition]

# helper functions

def open_import_tracking(IN_tracking_file: str) -> pd.DataFrame:
    """
    Check if the import tracking data file with a list of imported files and their UUIDs exists.
    
    Parameters
    ----------
    IN_tracking_file : str
        The path to the import tracking data file.

    Returns
    -------
    pd.DataFrame
        A Pandas DataFrame representing the import tracking data. If the file exists,
        it is loaded from the file; otherwise, an empty DataFrame is created.
        
    Example
    -------
    To check and load the import tracking data from a file:
    
    >>> import_tracking_file_path = "import_tracking.csv"
    >>> import_data = open_import_tracking(import_tracking_file_path)
    >>> print(import_data.head())

    Notes
    -----
    If the import tracking file does not exist, a new empty DataFrame with columns
    'uuid', 'file_path', and 'type of file' is created. Subsequent log entries
    will indicate the file's existence or creation.

    """
    if os.path.isfile(IN_tracking_file):
        logging.info("Loading import tracking file <%s>.", IN_tracking_file)
        return pd.read_csv(IN_tracking_file)
    else:
        logging.error("Import tracking file <%s> is missing. Creating a new empty file.", IN_tracking_file)
        columns = ["uuid", "file_path", "type of file", "import status"]
        return pd.DataFrame(columns=columns)



def get_all_file_paths(directory_path):
    """
    Retrieve a list of all file paths within a directory and its subdirectories.

    Parameters
    ----------
    directory_path : str
        The path to the directory for which file paths need to be collected.

    Returns
    -------
    list of str
        A list containing the absolute file paths of all files found within
        the specified directory and its subdirectories.

    Example
    -------
    To collect all file paths in a directory and its subdirectories:

    >>> directory = "/path/to/directory"
    >>> files = get_all_file_paths(directory)
    >>> print(files)
    ['/path/to/directory/file1.txt', '/path/to/directory/subdir/file2.pdf', ...]

    """

    file_paths = []
    for root, _, files in os.walk(directory_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)
    return file_paths


# generate an uuid for a document
def get_uuid5(IN_file):
    unique_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, IN_file))
    return(unique_id)



def generate_import_list(in_source_directory: str):
    file_info = []
    for root, _, files in os.walk(in_source_directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_uuid = get_uuid5(file_path)
            file_type = type(file_path)
            file_info.append({"File Path": file_path,
                              "UUID": file_uuid,
                              "Type of File": file_type,
                              "Status": "Not Processed"  # Initial status
                              })
    df_result = pd.DataFrame(file_info)
    return df_result

# determine available loaders based on their extension
available_loader_types = [ext.lstrip(".") for ext in LOADER_MAPPING.keys()]


def main_execution():
    # open panda dataframe with list of already loaded documents. if not existing, establish pd dataframe
	import_tracking_df = open_import_tracking(IN_tracking_file= IMPORT_LOG_FILE)
    
	# generate a list of all documents in import dir
	tmp_import_df = generate_import_list(in_source_directory="/Users/swmoeller/python/2023/large_language_model/BrainGPT/data/10_raw")
	print(tmp_import_df)
    
	# Case
	# 1) document existiert in import-liste in same folder
	#    --> log Eintrag, no further action

	# 2) same document exist, but in different folder
	#    --> Status Eintrag in import-log: file existiert in anderen Verzeichnis
	
	# 3) document does not exist, but no loader available
	#    --> Status Eintrag in import-log: missing importer

	# 4) document does not exist, loader is available
	#    --> new enty in import log with status: loaded
	 	# load the document

		# convert it into text

		# chunk it

        # save it to vectorstore


# ===================  
# MAIN  
# ===================

if __name__ == "__main__":
    # Code that runs when the script is executed directly
    main_execution()
    logging.info("End of script execution\n\n")

