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

import logging
import os # Operating System module for interacting with the OS environment
import uuid # Universally Unique Identifier (UUID) module for generating unique identifiers

from typing import List # Typing module for defining type hints
from dotenv import load_dotenv # Dotenv library for loading environment variables from a file

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
                    format = "%(asctime)s : %(levelname)s : Line No. : %(lineno)d - %(message)s",
                    filename = 'braingpt.log',
                    filemode = 'a')
logging.info("Start of script execution")


# [IMPORTS of environment constants]
load_dotenv()   # load environmental variables
ROOT_DIR = "/Users/swmoeller/python/2023/large_language_model/BrainGPT"
# // TODO - using function from utilities_braingpt, automatize identification of root-directory

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

DOC2SCAN_DATA_DIR = "/Users/swmoeller/python/2023/large_language_model/BrainGPT/data/10_raw"



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
        logging.info("Loading import tracking file <%s> into panda dataframe.", IN_tracking_file)
        return pd.read_csv(IN_tracking_file)
    else:
        logging.error("Import tracking file <%s> is missing. Creating a new panda dataframe.", IN_tracking_file)
        columns = ["File Name", "File Path", "UUID", "File Extension", "Type of File", "Status"]
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
    return unique_id



def generate_import_list(in_source_directory: str):
    file_info = []
    for root, _, files in os.walk(in_source_directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_name = os.path.basename(file_path)
            file_extension = os.path.splitext(file_name)[1]
            file_uuid = get_uuid5(file_name)
            file_type = type(file_path)
            file_info.append({"File Name" : file_name,
                              "File Path": file_path,
                              "UUID": file_uuid,
                              "File Extension": file_extension,
                              "Type of File": file_type,
                              "Status": "not processed"  # Initial status
                              })
    df_result = pd.DataFrame(file_info)
    return df_result


def merge_dataframes(IN_main_df, IN_tmp_df):
    """
    Merge two DataFrames based on unique combinations of "File Path" and "UUID" columns.

    This function takes two input DataFrames, IN_main_df and IN_tmp_df, and combines them
    into a single DataFrame. It handles two distinct cases:

    1. If a row with the same "UUID" exists in IN_main_df but has a different "File Path",
       the "Status" for the matching rows is set to "duplicate?".

    2. If a row with the same "UUID" and "File Path" combination does not exist in IN_main_df,
       a new row is added with "Status" set to "import".

    Parameters:
    - IN_main_df (DataFrame): The main DataFrame to which the data will be merged.
    - IN_tmp_df (DataFrame): The temporary DataFrame containing additional data to merge.

    Returns:
    DataFrame: A new DataFrame containing the merged data.

    Example:
    ```python
    # Sample DataFrames
    main_df = pd.DataFrame({'File Path': ['path1', 'path2'], 'UUID': ['uuid1', 'uuid2']})
    tmp_df = pd.DataFrame({'File Path': ['path2', 'path3'], 'UUID': ['uuid2', 'uuid3']})

    # Merge the DataFrames
    result_df = merge_dataframes(main_df, tmp_df)

    # Resulting DataFrame will contain unique rows from main_df and tmp_df with appropriate "Status" values.
    ```
    """
    df1 = IN_main_df.copy()
    df2 = IN_tmp_df

    # Identify rows with the same UUID but different File Path in df1
    duplicate_rows = df1[df1.duplicated(subset=['UUID'], keep=False)]

    # Filter DF2 for unique combinations of "File Path" and "UUID" not in DF1
    unique_rows_df2 = df2[~df2.set_index(["File Path", "UUID"]).index.isin(df1.set_index(["File Path", "UUID"]).index)]

    # Set the "Status" column for the filtered rows to "import"
    unique_rows_df2["Status"] = "import"

    # Set the "Status" column for duplicate rows to "duplicate?"
    df1.loc[duplicate_rows.index, "Status"] = "duplicate?"

    # Concatenate DF1 and the filtered DF2
    combined_df = pd.concat([df1, unique_rows_df2], ignore_index=True)

    logging.info("Pandas Dataframe updated with files 'to be imported' or 'duplicated'.")
    return combined_df


def get_available_loader_types(loader_mapping):
    """
    Get available loader types based on their extension from a loader mapping dictionary.

     Parameters:
    - loader_mapping (dict): A dictionary mapping file extensions to loader types.

    Returns:
    list: A list of available loader types.
    """
    # available_loader_types = [ext.lstrip(".") for ext in loader_mapping.keys()]
    available_loader_types = list(loader_mapping.keys())
    logging.info("Possible file.extensions for document loader analyzed and list of extensions generated.")

    return available_loader_types


def split_doc_into_chunks(IN_document):

    chunks = []
    if len(IN_document) != 0:
        # Split the documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        chunks = text_splitter.split_documents(IN_document)

    else:
        logging.error("No document found for splitting.")
        
    return(chunks)


def process_document_list(IN_file_list, IN_valid_extensions):
    """
    Process each row of the DataFrame and update the 'Status' column accordingly.

    Parameters:
    - df (DataFrame): The DataFrame containing the data to be processed.

    Returns:
    DataFrame: The DataFrame with updated 'Status' values.
    """
    text_chunks = []
    for index, row in IN_file_list.iterrows():
        if row['Status'] == "import":
            file_path = row["File Path"]
            file_name = row["File Name"]
            file_extension = row["File Extension"]

            if file_extension in IN_valid_extensions:
                # Load the document
                loader_class, loader_args = LOADER_MAPPING[file_extension]
                loader = loader_class(file_path, **loader_args)
                document = loader.load()
                logging.info("Document %s loaded.", file_name)
                text_chunks= split_doc_into_chunks(document)
                logging.info("Splitting of document <%s> done with <%s> chunks",file_name, len(text_chunks))
               
                # Perform processing steps here (e.g., load the file)
                # After processing, update the 'Status' to 'imported'
               
                IN_file_list.at[index, 'Status'] = 'imported'

            else:
                IN_file_list.at[index, 'Status'] = 'no loader available'
                logging.error("No document loader found!")

    return IN_file_list, text_chunks



def existence_vectorstore(IN_datastore_location: str) -> bool:
    """
    Checks if vectorstore exists
    """
    if os.path.exists(os.path.join(IN_datastore_location, 'index')):
        if os.path.exists(os.path.join(IN_datastore_location, 'chroma-collections.parquet')) and os.path.exists(os.path.join(IN_datastore_location, 'chroma-embeddings.parquet')):
            list_index_files = os.path.join(IN_datastore_location, 'index/*.bin')
            list_index_files += os.path.join(IN_datastore_location, 'index/*.pkl')
            # At least 3 documents are needed in a working vectorstore
            if len(list_index_files) > 3:
                return True
    return False



def update_vectorstore(IN_datastore_location: str, IN_embedding_function, IN_chromadb_setting, IN_text_chunks):
    if existence_vectorstore(IN_datastore_location=IN_datastore_location) is True:

        # loading the vectorstore
        vectordb = Chroma(persist_directory=IN_datastore_location, 
                        embedding_function=IN_embedding_function,
                        client_settings=IN_chromadb_setting)
        logging.info("Vectorstore loaded from loction <%s>", IN_datastore_location)


        # adding documents
        vectordb.add_documents(IN_text_chunks)
        logging.info("In total, %s file chunks were imported into the vectorstore.", len(text_chunks))

    else:
        logging.error("No vectorstore exists. New store established in <%s>.", IN_datastore_location)
        vectordb = Chroma.from_documents(documents=IN_text_chunks, 
                                embedding=IN_embedding_function, 
                                persist_directory=IN_datastore_location, 
                                client_settings=IN_chromadb_setting)

    # saving the vectorstore
    vectordb.persist()


def main_execution():
    # open pd dataframe with list of already loaded documents. if not existing, establish pd dataframe
    import_tracking_df = open_import_tracking(IN_tracking_file=IMPORT_LOG_FILE)

	# generate a list of all documents in import dir
    tmp_import_df = generate_import_list(in_source_directory=DOC2SCAN_DATA_DIR)

    # match both dataframes and mark sets to be imported with "import" or "duplicate"
    document_list_df = merge_dataframes(import_tracking_df, tmp_import_df)

    # generate the list of possible extensions from the document loader
    valid_extension = get_available_loader_types(loader_mapping=LOADER_MAPPING)


    # process the document_list and import/flag the documents
    document_list_df, text_chunks = process_document_list(IN_file_list=document_list_df,
                                                          IN_valid_extensions=valid_extension)

    # initializing the vectorstore
    embeddings = OpenAIEmbeddings()

    logging.info("In total, %s file chunks due to be imported into the vectorstore.", len(text_chunks))
    update_vectorstore(IN_datastore_location=PROCESSED_DATA_DIR,
                       IN_embedding_function= embeddings,
                       IN_chromadb_setting=CHROMA_SETTINGS,
                       IN_text_chunks=text_chunks)

  
    document_list_df.to_csv(IMPORT_LOG_FILE, index=False)
    logging.info("Document import list saved.")


# ===================
# MAIN
# ===================

if __name__ == "__main__":
    # Code that runs when the script is executed directly
    main_execution()
    logging.info("End of script execution\n\n")
