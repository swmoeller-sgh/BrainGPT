#!/usr/bin/env python
""" 
[Purpose]
=========
Ingestion of text documents provided within one directory. 
Duplicates are checked and excluded. Imported documents are maintained in an import log.

[References]
============



"""

# [IMPORT of modules and packages]

import logging
import os # Operating System module for interacting with the OS environment
import uuid # Universally Unique Identifier (UUID) module for generating unique identifiers
import time

from dotenv import load_dotenv, find_dotenv # Dotenv library for loading env variables from a file

import openai # OpenAI library for working with OpenAI's GPT-3 or other models

import pandas as pd # Pandas library for data manipulation and analysis

# Custom module imports related to langchain, a specific library or framework
from langchain.document_loaders import TextLoader  # Text document loading for langchain
from langchain.vectorstores import Chroma  # Vector storage for langchain
from langchain.embeddings import OpenAIEmbeddings  # Embeddings for langchain
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Text splitting for langchain


#pylint: disable=W0611, W0404
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

#pylint: disable=E0401
from brain_gpt_config import config_braingpt # type: ignore

# [INITIALIZE environment]
env_file = find_dotenv(".env")
load_dotenv(env_file)

# [IMPORTS of environment constants]
openai.api_key = os.getenv("OPENAI_API_KEY")

# [CORE directories]
ROOT_DIR = os.path.join(str(os.getenv("ROOT_DIR"))) # Define the root directory for file operations
# // TODO - using function from utilities_braingpt, automatize identification of root-directory

BRAINGPT_INQUIRY_LOG_FILE = os.path.join(ROOT_DIR,  # Define the path to the query log file
                                         str(os.getenv("LOG_DIR")),
                                         str(os.getenv("INGEST_LOG_NAME")))


# [LOGGING settings]
config_braingpt.setup_logging(in__log_path_file= BRAINGPT_INQUIRY_LOG_FILE)
logging.info("Start of script execution: %s", os.path.basename(__file__))


# [DIRECTORY setup]
PROCESSED_DATA_DIR = os.path.join(ROOT_DIR,         # storage of ChromaDB files
                                  str(os.getenv("PROCESSED_DATA_DIR")),
                                  str(os.getenv("CHROMA_DB")))
logging.info("Target directory for processed files: %s", PROCESSED_DATA_DIR)

IMPORT_LOG_FILE = os.path.join(ROOT_DIR,            # Define the path to the ingestion log file
                               str(os.getenv("LOG_DIR")),
                               str(os.getenv("IMPORT_LOG_NAME")))
logging.info("Log file on imported documents: %s", IMPORT_LOG_FILE)

DOC2SCAN_DATA_DIR = os.path.join(ROOT_DIR,          # Path to folder for documents to be scanned
                                 str(os.getenv("DOC2SCAN_DATA_DIR")))
logging.info("Source directory for documents: %s\n", DOC2SCAN_DATA_DIR)


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

def open_import_tracking(in__tracking_file: str) -> pd.DataFrame:
    """
    Check if the import tracking data file with a list of imported files and their UUIDs exists.
    
    Parameters
    ----------
    in__tracking_file : str
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
    logging.info("[INFO] Start opening import tracking file")
    if os.path.isfile(in__tracking_file):
        logging.info("Tracking file <%s> loaded into panda dataframe.\n", in__tracking_file)
        return pd.read_csv(in__tracking_file)
    else:
        logging.warning("Tracking file <%s> is missing. Creating a new panda dataframe.\n",
                        in__tracking_file)
        columns = ["File Name", "File Path", "UUID", "File Extension", "Type of File", "Status"]
        return pd.DataFrame(columns=columns)


def get_all_file_paths(in__directory_path):
    """
    Retrieve a list of all file paths within a directory and its subdirectories.

    Parameters
    ----------
    in__directory_path : str
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
    logging.info("[INFO] Recording all files from all pathes.")
    file_paths = []
    for root, _, files in os.walk(in__directory_path):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)
    logging.info("# %s files counted in folders.\n", len(file_paths))

    return file_paths


def get_uuid5(in__file):
    """
    Generate a UUID (Universally Unique Identifier) based on the given input file.

    This function generates a UUID version 5 based on the DNS namespace and the provided
    input file name. UUIDs are unique identifiers used for various purposes, and version 5
    UUIDs are generated using a namespace and a name.

    Parameters:
    - in__file (str): The input file name for which the UUID will be generated.

    Returns:
    str: A unique UUID version 5 string based on the provided file name.

    Example:
    ```python
    # Generate a UUID based on a file name
    file_name = "example.txt"
    uuid = get_uuid5(file_name)
    print(uuid)
    ```

    Notes:
    - The function uses the DNS namespace for generating UUIDs, which ensures uniqueness.

    """
    unique_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, in__file))
    return unique_id


def generate_import_list(in__source_directory: str) -> pd.DataFrame:
    """
    Generate a Pandas DataFrame of file information from files in the specified directory.

    This function traverses the specified directory and its subdirectories to collect information
    about files, such as file name, file path, UUID, file extension, file type, and an initial
    status of "not processed" for each file. It returns a Pandas DataFrame containing 
    this information.

    Parameters:
    - in__source_directory (str): The path to the directory to scan for files.

    Returns:
    pd.DataFrame: A Pandas DataFrame containing file information.

    Example:
    ```python
    # Generate a DataFrame from files in the 'my_files' directory
    file_info_df = generate_import_list('/path/to/my_files')
    print(file_info_df.head())
    ```

    Notes:
    - The function collects information about files found in the specified directory 
      and its subdirectories.
    - It assigns an initial status of "not processed" to each file.

    """
    file_info = []  # Initialize an empty list to store file information
    count = 0  # Initialize a counter for the number of files found

    # Log information about the operation
    logging.info("[INFO] Generating a Pandas DataFrame from files in the import directory.")

    # Traverse the specified directory and its subdirectories
    for root, _, files in os.walk(in__source_directory):
        for file in files:
            file_path = os.path.join(root, file)  # Get the full file path
            file_name = os.path.basename(file_path)  # Extract the file name
            file_extension = os.path.splitext(file_name)[1]  # Extract the file extension
            file_uuid = get_uuid5(file_name)  # Generate a UUID for the file
            file_type = type(file_path)  # Determine the type of file (not the correct way,
                                         # consider using `magic` library)
            count += 1  # Increment the file count

            # Append file information as a dictionary to the list
            file_info.append({
                "File Name": file_name,
                "File Path": file_path,
                "UUID": file_uuid,
                "File Extension": file_extension,
                "Type of File": file_type,
                "Status": "not processed"  # Initial status
            })

    # Log the number of documents found and the next step
    logging.info("%s documents found in the import directory. Next: Checking import status.\n",
                 count)

    # Create a Pandas DataFrame from the collected file information
    df_result = pd.DataFrame(file_info)

    return df_result  # Return the DataFrame containing file information


def merge_dataframes(in__main_df, in__tmp_df):
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

    # Resulting DataFrame will contain unique rows from main_df and tmp_df with appropriate 
    "Status" values.
    ```
    """
    df1 = in__main_df.copy()
    df2 = in__tmp_df
    logging.info("[INFO] Validating import status of found files in import directory.")
    # Identify rows with the same UUID but different File Path in df1
    duplicate_rows = df1[df1.duplicated(subset=['UUID'], keep=False)]

    # Filter DF2 for unique combinations of "File Path" and "UUID" not in DF1
    # Step 1: Set the index for df2
    index_df2 = df2.set_index(["File Path", "UUID"])

    # Step 2: Set the index for df1
    index_df1 = df1.set_index(["File Path", "UUID"])

    # Step 3: Check which index values in df2 are not in df1
    unique_rows_mask = ~index_df2.index.isin(index_df1.index)

    # Step 4: Use the mask to filter df2 and get unique rows
    unique_rows_df2 = df2[unique_rows_mask]


    # Set the "Status" column for the filtered rows to "import"
    unique_rows_df2["Status"] = "import"

    # Set the "Status" column for duplicate rows to "duplicate?"
    df1.loc[duplicate_rows.index, "Status"] = "duplicate?"

    # Concatenate DF1 and the filtered DF2
    combined_df = pd.concat([df1, unique_rows_df2], ignore_index=True)

    logging.info("Dataframe updated with %s files, status set: 'to be imported' or 'duplicated'.\n",
                 len(unique_rows_df2))
    return combined_df


def get_available_loader_types(in__loader_mapping):
    """
    Get available loader types based on their extension from a loader mapping dictionary.

     Parameters:
    - loader_mapping (dict): A dictionary mapping file extensions to loader types.

    Returns:
    list: A list of available loader types.
    """
    # available_loader_types = [ext.lstrip(".") for ext in loader_mapping.keys()]
    logging.info("[INFO] Generating & Updating list of available document loader.")
    available_loader_types = list(in__loader_mapping.keys())
    logging.info("Possible document loader analyzed and list of extensions generated\n")

    return available_loader_types


def split_doc_into_chunks(in__document):
    """
    Split a document into chunks of text.

    This function takes a document as input and splits it into smaller chunks of text.
    It uses the RecursiveCharacterTextSplitter from the langchain library for this purpose.

    Parameters:
    - IN_document: The document to be split into chunks.

    Returns:
    list: A list of text chunks.

    Example:
    ```python
    # Split a document into chunks
    document_text = "This is a long document text..."
    chunks = split_doc_into_chunks(document_text)
    print(chunks)
    ```

    Notes:
    - Function uses the RecursiveCharacterTextSplitter to divide the document into smaller pieces.
    - It returns a list of text chunks.

    """
    chunks = []  # Initialize an empty list to store text chunks

    if len(in__document) != 0:
        # Split the document into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        chunks = text_splitter.split_documents(in__document)
    else:
        # Log an error if no document is provided for splitting
        logging.error("No document found for splitting.\n")

    return chunks  # Return the list of text chunks


def process_document_list(in__file_list, in__valid_extensions):
    """
    Process each row of the DataFrame and update the 'Status' column accordingly.

    This function iterates through each row of the input DataFrame and processes the documents
    based on their file extension. It loads the document, splits it into text chunks, and updates
    the 'Status' column. If a document has a valid loader, it is marked as 'imported'. Otherwise,
    it is marked as 'no loader available'.

    Parameters:
    - IN_file_list (DataFrame): The DataFrame containing the data to be processed.
    - IN_valid_extensions (list): A list of valid file extensions for document loading.

    Returns:
    DataFrame: The DataFrame with updated 'Status' values.
    list: A list of text chunks extracted from the processed documents.

    Example:
    ```python
    # Process a DataFrame of documents and update their status
    document_df = pd.DataFrame({'File Path': ['file1.pdf', 'file2.txt'], 
                  'Status': ['import', 'import']})
    valid_extensions = ['.pdf', '.txt']
    result_df, text_chunks = process_document_list(document_df, valid_extensions)
    ```

    Notes:
    - This function updates the 'Status' column of the DataFrame based on the processing result.
    - It returns the updated DataFrame and a list of text chunks from the processed documents.

    """
    text_chunks = []  # Initialize an empty list to store text chunks
    total_chunks = []  # Initialize an empty list to store total text chunks

    for index, row in in__file_list.iterrows():
        if row['Status'] == "import":
            file_path = row["File Path"]
            file_name = row["File Name"]
            file_extension = row["File Extension"]

            if file_extension in in__valid_extensions:
                # Load the document
                loader_class, loader_args = LOADER_MAPPING[file_extension]
                loader = loader_class(file_path, **loader_args)
                document = loader.load()
                logging.info("Document %s loaded.", file_name)

                # Split the document into text chunks
                text_chunks = split_doc_into_chunks(document)
                total_chunks = total_chunks + text_chunks
                logging.info("Split of document <%s> done w/ <%s> chunks. Running total %s chunks.",
                             file_name, len(text_chunks), len(total_chunks))

                # Perform processing steps here (e.g., load the file)
                # After processing, update the 'Status' to 'imported'
                in__file_list.at[index, 'Status'] = 'imported'
            else:
                # Mark the document as 'no loader available' if the extension is not valid
                in__file_list.at[index, 'Status'] = 'no loader available'
                logging.error("No document loader found!")

    return in__file_list, text_chunks  # Return the updated DataFrame and the list of text chunks


def existence_vectorstore(in__datastore_location: str) -> bool:
    """
    Check if a vector store exists at the specified location.

    This function checks if a vector store exists at the given directory location. A valid 
    vector store must contain the 'index' directory, 'chroma-collections.parquet', 
    'chroma-embeddings.parquet', and at least three index files ('*.bin' or '*.pkl').

    Parameters:
    - IN_datastore_location (str): The path to the vector store directory to check.

    Returns:
    bool: True if a valid vector store exists, False otherwise.

    Example:
    ```python
    # Check if a vector store exists at the specified location
    store_location = "/path/to/vectorstore"
    if existence_vectorstore(store_location):
        print("Vector store exists.")
    else:
        print("Vector store does not exist.")
    ```

    Notes:
    - A valid vector store must meet the specific directory and file requirements to be considered
      as existing.

    """
    logging.info("[INFO] Checking if vector store exists.")

    # Check if the 'index' directory exists
    if os.path.exists(os.path.join(in__datastore_location, 'index')):
        # Check if 'chroma-collections.parquet' and 'chroma-embeddings.parquet' exist
        # Step 1: Create the file paths
        collections_file_path = os.path.join(in__datastore_location, 'chroma-collections.parquet')
        embeddings_file_path = os.path.join(in__datastore_location, 'chroma-embeddings.parquet')

        # Step 2: Check if both files exist
        if os.path.exists(collections_file_path) and os.path.exists(embeddings_file_path):
            # List index files ('*.bin' and '*.pkl')
            list_index_files = os.path.join(in__datastore_location, 'index/*.bin')
            list_index_files += os.path.join(in__datastore_location, 'index/*.pkl')

            # Check if there are at least three index files
            if len(list_index_files) > 3:
                logging.info("Vector store exists at %s.\n", in__datastore_location)
                return True

    # If the vector store doesn't meet the criteria, log a warning and return False
    logging.warning("Vector store does not exist or is incomplete at %s.\n", in__datastore_location)
    return False


def update_vectorstore(in__datastore_location: str,
                       in__embedding_function,
                       in__chromadb_setting,
                       in__text_chunks):
    """
    Update a vector store with new text chunks.

    This function updates an existing vector store or creates a new one if it doesn't exist. 
    It loads the vector store, adds the provided text chunks, and saves the updated vector store.

    Parameters:
    - IN_datastore_location (str): The path to the vector store directory.
    - IN_embedding_function (callable): The embedding function to use for text chunks.
    - IN_chromadb_setting (Settings): The ChromaDB settings for the vector store.
    - IN_text_chunks (list of str): A list of text chunks to add to the vector store.

    Returns:
    None

    Example:
    ```python
    # Define vector store settings and update the vector store
    datastore_location = "/path/to/vectorstore"
    embedding_func = my_embedding_function
    chromadb_settings = Settings(chroma_db_impl='duckdb+parquet')
    text_chunks = ["chunk1", "chunk2", "chunk3"]
    update_vectorstore(datastore_location, embedding_func, chromadb_settings, text_chunks)
    ```

    Notes:
    - This function first checks if a vector store exists at the specified location using the
      `existence_vectorstore` function.
    - If a vector store exists, it loads the store, adds text chunks, and saves the updated store.
    - If no vector store exists, it creates a new vector store, adds the text chunks, and saves it.

    """
    if existence_vectorstore(in__datastore_location=in__datastore_location) is True:

        # Loading the existing vector store
        vectordb = Chroma(
            persist_directory=in__datastore_location,
            embedding_function=in__embedding_function,
            client_settings=in__chromadb_setting
        )
        logging.info("Vector store loaded from location <%s>", in__datastore_location)

        # Adding the provided text chunks to the vector store
        vectordb.add_documents(in__text_chunks)
        logging.info("A total of %s file chunks were imported into the vector store.\n",
                     len(in__text_chunks))

    else:
        logging.error("No vector store exists. Creating a new store in <%s>.",
                      in__datastore_location)

        # Creating a new vector store and adding the text chunks
        vectordb = Chroma.from_documents(
            documents=in__text_chunks,
            embedding=in__embedding_function,
            persist_directory=in__datastore_location,
            client_settings=in__chromadb_setting
        )
        logging.info("A total of %s file chunks were imported into the vector store.\n",
                     len(in__text_chunks))

    # Saving the updated vector store (currently commented out)
    # vectordb.persist()
    vectordb = None

#pylint: disable=W0511
# FIXME Below function

#pylint: disable=W0105
'''
def update_vectorstore_collection(in__datastore_location: str,
                                  in__embedding_function,
                                  in__chromadb_setting,
                                  in__text_chunks,
                                  group_by_attribute="collection"):
    """
    Update a vector store with new text chunks grouped into collections.

    This function updates an existing vector store or creates a new one if it doesn't exist.
    It loads the vector store, groups the provided text chunks into collections based on a
    specified attribute, and adds the collections to the vector store.

    Parameters:
    - IN_datastore_location (str): The path to the vector store directory.
    - IN_embedding_function (callable): The embedding function to use for text chunks.
    - IN_chromadb_setting (Settings): The ChromaDB settings for the vector store.
    - IN_text_chunks (list of dict): A list of text chunks and their associated attributes.
    - group_by_attribute (str): The attribute by which to group the documents into collections.

    Returns:
    None

    Example:
    ```python
    # Define vector store settings and update the vector store with grouped collections
    datastore_location = "/path/to/vectorstore"
    embedding_func = my_embedding_function
    chromadb_settings = Settings(chroma_db_impl='duckdb+parquet')
    text_chunks = [
        {"text": "chunk1", "collection": "collection1"},
        {"text": "chunk2", "collection": "collection1"},
        {"text": "chunk3", "collection": "collection2"}
    ]
    update_vectorstore(datastore_location, embedding_func, chromadb_settings, text_chunks)
    ```

    Notes:
    - This function groups the text chunks into collections based on the 
      specified `group_by_attribute`.
    - It then adds the collections to the vector store instead of individual documents.
    """
    if existence_vectorstore(in__datastore_location=in__datastore_location) is True:
        # Loading the existing vector store
        vectordb = Chroma(
            persist_directory=in__datastore_location,
            embedding_function=in__embedding_function,
            client_settings=in__chromadb_setting
        )
        logging.info("Vector store loaded from location <%s>", in__datastore_location)

        # Group text chunks into collections based on the specified attribute
        collections = {}
        for chunk in in__text_chunks:
            attribute_value = chunk.get(group_by_attribute, "default")
            if attribute_value not in collections:
                collections[attribute_value] = []
            collections[attribute_value].append(chunk["text"])

        # Add collections to the vector store
        for collection_name, collection_text_chunks in collections.items():
            # Create a single document from the chunks in the collection
            collection_text = "\n".join(collection_text_chunks)

            # Add the collection to the vector store
            vectordb.add_documents([collection_text], attribute_name=group_by_attribute, attribute_value=collection_name)
            logging.info("Collection '%s' with %s text chunks imported into the vector store.\n",
                         collection_name, len(collection_text_chunks))

    else:
        logging.error("No vector store exists. Creating a new store in <%s>.",
                      in__datastore_location)

        # Creating a new vector store and adding collections
        vectordb = Chroma(
            persist_directory=in__datastore_location,
            embedding_function=in__embedding_function,
            client_settings=in__chromadb_setting
        )

        # Group text chunks into collections based on the specified attribute
        collections = {}
        for chunk in in__text_chunks:
            attribute_value = chunk.get(group_by_attribute, "default")
            if attribute_value not in collections:
                collections[attribute_value] = []
            collections[attribute_value].append(chunk["text"])

        # Add collections to the vector store
        for collection_name, collection_text_chunks in collections.items():
            # Create a single document from the chunks in the collection
            collection_text = "\n".join(collection_text_chunks)
            # Add the collection to the vector store

            vectordb.add_documents(collection_text, 
                                   attribute_name=group_by_attribute, 
                                   attribute_value=collection_name)
            logging.info("Collection '%s' with %s text chunks imported into the vector store.\n",
                         collection_name, len(collection_text_chunks))

    # Saving the updated vector store (currently commented out)
    # vectordb.persist()
    vectordb = None
'''

def main_execution():
    """
    Main execution function for importing and processing documents.

    This function performs the following steps:
    1. Opens an existing import tracking DataFrame or creates a new one.
    2. Generates a list of all documents in the import directory.
    3. Matches both DataFrames and marks documents to be imported with "import" or "duplicate."
    4. Generates a list of valid extensions from the document loader.
    5. Processes the document list and imports/flags the documents.
    6. Initializes the vector store using OpenAI embeddings.
    7. Updates the vector store with the imported text chunks.
    8. Saves the document import list to a CSV file.

    Parameters:
    None

    Returns:
    None

    Example:
    ```python
    # Execute the main script
    main_execution()
    ```

    Notes:
    - This function orchestrates the entire document import and processing workflow.
    - It ensures that existing documents are not imported again and handles the import status.
    - After importing documents, it updates the vector store with the text chunks.

    """
    # Open a Pandas DataFrame with a list of already loaded documents.
    # If it doesn't exist, establish a new DataFrame.
    import_tracking_df = open_import_tracking(in__tracking_file=IMPORT_LOG_FILE)

    # Generate a list of all documents in the import directory.
    tmp_import_df = generate_import_list(in__source_directory=DOC2SCAN_DATA_DIR)

    if len(tmp_import_df) == 0:
        logging.info("No documents found in the import directory. Exiting.\n")
        return

    # Match both DataFrames and mark documents to be imported with "import" or "duplicate."
    document_list_df = merge_dataframes(import_tracking_df, tmp_import_df)

    # Generate the list of possible extensions from the document loader.
    valid_extension = get_available_loader_types(in__loader_mapping=LOADER_MAPPING)

    # Process the document list and import/flag the documents.
    document_list_df, text_chunks = process_document_list(
        in__file_list=document_list_df,
        in__valid_extensions=valid_extension
    )

    # Initializing the vector store using OpenAI embeddings.
    embeddings = OpenAIEmbeddings()             # type: ignore

    # Update the vector store with the imported text chunks.
    update_vectorstore(
        in__datastore_location=PROCESSED_DATA_DIR,
        in__embedding_function=embeddings,
        in__chromadb_setting=CHROMA_SETTINGS,
        in__text_chunks=text_chunks
    )

    # Save the document import list to a CSV file.
    document_list_df.to_csv(IMPORT_LOG_FILE, index=False)
    logging.info("Document import list saved.")


# ===================
# MAIN
# ===================

if __name__ == "__main__":
    # Code that runs when the script is executed directly
    main_execution()
    time.sleep(0.5)
    logging.info("End of script execution\n\n")
