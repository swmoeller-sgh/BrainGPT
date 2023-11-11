""" Module providing question - answer functions using an existing vectorstore"""

# --------------------------------
# [ERROR HANDLING]
#pylint: disable=E0401
# --------------------------------

# --------------------------------
# [IMPORT of modules and packages]
# --------------------------------
import os
import logging
import pickle

import openai

from dotenv import load_dotenv, find_dotenv

# // TODO Neuinstallation env: langchain.vectorstores --> langchain.vectorstores.chroma wechseln
from langchain.llms.openai import OpenAI
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

from chromadb.config import Settings

from braingpt_config import config_braingpt # type: ignore


# --------------------------------
# [INITIALIZE environment]
# --------------------------------
env_file = find_dotenv(".env")
load_dotenv(env_file)


# --------------------------------
# [DIRECTORY setup]
# --------------------------------
ROOT_DIR = os.path.join(str(os.getenv("ROOT_DIR"))) # Define the root directory for file operations
BRAINGPT_QA_LOG = os.path.join(ROOT_DIR,  # Define the path to the query log file
                                str(os.getenv("LOG_DIR")),
                                str(os.getenv("QUERY_LOG_NAME")))
PROCESSED_DATA_DIR = os.path.join(ROOT_DIR,
                                  str(os.getenv("PROCESSED_DATA_DIR")),
                                  str(os.getenv("CHROMA_DB")))
FILEPATH_DEBUGANSWER_TXT = "data/10_raw/qa_chained_history_sample_answer.txt"
FILEPATH_DEBUGANSWER_PKL = "data/10_raw/qa_chained_history_sample_answer.pkl"


# --------------------------------
# [LOGGING initial state]
# --------------------------------
config_braingpt.setup_logging(in__log_path_file= BRAINGPT_QA_LOG)   # [LOGGING settings]

logging.info("Initializing script: %s", os.path.basename(__file__))
logging.info("Target directory for processed files: %s", PROCESSED_DATA_DIR)


# --------------------------------
# [CONFIGURATION]
# --------------------------------
openai.api_key = os.getenv("OPENAI_API_KEY")                        # openAI-settings
USED_MODEL = "OPENAI"

embeddings = OpenAIEmbeddings() # type: ignore
CHROMA_SETTINGS = Settings(                                         # Define the Chroma settings
        persist_directory=PROCESSED_DATA_DIR,
        anonymized_telemetry=False
)
vectordb = Chroma(persist_directory=PROCESSED_DATA_DIR,
                  embedding_function=embeddings)


# --------------------------------
# [SUPPORTING FUNCTIONS] to load debug answers, decompose answers, etc.
# --------------------------------
def get_local_answer(in_filepath):
    """
    Load data from a local file based on the file extension.

    Parameters:
    in_filepath (str): The path to the input file.

    Returns:
    object: The data loaded from the file, or a message for unsupported file extensions.
    """

    logging.info("Local input data from file: %s", in_filepath)

    # Use os.path.basename to get the filename
    filename = os.path.basename(in_filepath)

    # Use os.path.splitext to separate the filename and extension
    file_extension = os.path.splitext(filename)[1]
    logging.info("Detected file extension: %s", file_extension)

    # Use a match statement to determine the file type based on its extension
    match file_extension:
        case ".pkl":
            try:
                with open(in_filepath, "rb") as file:
                    pkl_file = pickle.load(file)
                return pkl_file
            except FileNotFoundError:
                print(f"File '{in_filepath}' not found.")
                return None

        case ".txt":
            try:
                with open(in_filepath, "r", encoding="utf-8") as file:
                    txt_file = file.read()
                return txt_file
            except FileNotFoundError:
                print(f"File '{in_filepath}' not found.")
                return None

        case _:
            return "Unsupported file extension"


def qa_chained_history_decompose(in_qa_response):
    """
    Decompose a QA response into its components and print information.

    Parameters:
    - in_qa_response (dict): A dictionary containing QA response information.

    Returns:
    Tuple[str, str, List[Tuple[str, str]], str]: A tuple containing the question, answer,
                                               consolidated sources, and chat history.

    Example:
    >>> response = {
    ...     "question": "What is the meaning of life?",
    ...     "answer": "The meaning of life is 42.",
    ...     "source_documents": [
    ...         {"page_content": "Content from source A", "metadata": {"source": "Source A"}},
    ...         {"page_content": "Content from source B", "metadata": {"source": "Source B"}},
    ...     ],
    ...     "chat_history": "User: What is the meaning of life?\nBot: The meaning of life is 42.",
    ... }
    >>> result = qa_chained_history_decompose(response)
    >>> print(result)
    ('What is the meaning of life?', 'The meaning of life is 42.',
     [('Content from source A', 'Source A'), ('Content from source B', 'Source B')],
     'User: What is the meaning of life?\nBot: The meaning of life is 42.')
    """
    print(in_qa_response.keys())

    # Extracting question and answer
    qa_question = in_qa_response["question"]
    qa_answer = in_qa_response["answer"]

    print("\nQuestion raised: ", qa_question)
    print("\n Answer provided: ", qa_answer)

    # Extracting information from source documents
    qa_reference = []
    for source in in_qa_response["source_documents"]:
        print("\n\nType of source: ", type(source))
        qa_source_page_content = source.page_content[:250].replace("\n", "")
        print("Extracted source original: ", source.page_content[:250].replace("\n", ""))
        print("Extracted document: ", source.metadata["source"].replace("\n", ""))
        qa_source_file = source.metadata["source"].replace("\n", "")
        qa_reference.append((qa_source_page_content, qa_source_file))

    print("\nConsolidated sources: ", qa_reference)

    # Extracting chat history
    qa_history = in_qa_response["chat_history"]
    print("\n History chat: ", qa_history)

    return qa_question, qa_answer, qa_reference, qa_history



def recompose(in_dictionary:dict, in_question:str, in_answer:str, in_sources:list):
    """
    Add a new entry to the input dictionary with the provided question, answer, and sources.

    Parameters:
    - in_dictionary (dict): The input dictionary to which the new entry will be added.
    - question (str): The question for the entry.
    - answer (str): The answer for the entry.
    - sources (list): A list of dictionaries containing 'extract' and 'file_link' keys for sources.

    Returns:
    dict: The updated dictionary with a new entry.

    Example:
    >>> result_dict = {}
    >>> result_dict = recompose(result_dict, "This is a question", "this is the answer",
    ...                         [{"extract": "extract 1", "file_link": "sdsd\\sds\\ds"},
    ...                          {"extract": "extract 2", "file_link": "sdsd\\sds\\ds2"}])
    >>> print(result_dict)
    {1: {'question': 'This is a question', 'answer': 'this is the answer',
         'sources': [{'extract': 'extract 1', 'file_link': 'sdsd\\sds\\ds'},
                     {'extract': 'extract 2', 'file_link': 'sdsd\\sds\\ds2'}]}}

    """
    def generate_dictionary(_question, _answer, _sources):
        """
        Generate a dictionary with the provided question, answer, and sources.

        Parameters:
        - question (str): The question for the entry.
        - answer (str): The answer for the entry.
        - sources (list): A list of dict containing 'extract' and 'file_link' keys for sources.

        Returns:
        dict: The generated dictionary.

        Example:
        >>> generate_dictionary("Q", "A", [{"extract": "ex1", "file_link": "link1"},
        ...                                  {"extract": "ex2", "file_link": "link2"}])
        {'question': 'Q', 'answer': 'A',
         'sources': [{'extract': 'ex1', 'file_link': 'link1'},
                     {'extract': 'ex2', 'file_link': 'link2'}]}

        """
        source_list = []

        for extract, file_link in _sources:
            source_dict = {
                'extract': extract,
                'file_link': file_link
            }
            source_list.append(source_dict)


        return {
            'question': _question,
            'answer': _answer,
            'sources': source_list
        }

    def get_next_key_number(existing_keys):
        """
        Get the next available key number based on existing keys.

        Parameters:
        - existing_keys (list): List of existing keys.

        Returns:
        int: The next available key number.

        Example:
        >>> get_next_key_number([1, 2, 4])
        5

        """
        if not existing_keys:
            return 1
        return max(existing_keys) + 1

    key = get_next_key_number(in_dictionary.keys())
    in_dictionary[key] = generate_dictionary(in_question,
                                             in_answer,
                                             in_sources)
    return in_dictionary

def print_recomposed_dictionary(in_dictionary):
    """
    Display the content of a recomposed dictionary.

    Parameters:
    - in_dictionary (dict): The dictionary to be displayed.

    Returns:
    None

    Example:
    >>> result_dict = {1: {'question': 'Q1', 'answer': 'A1',
    ...                     'sources': [{'extract': 'ex1', 'file_link': 'link1'},
    ...                                 {'extract': 'ex2', 'file_link': 'link2'}]},
    ...                2: {'question': 'Q2', 'answer': 'A2',
    ...                    'sources': [{'extract': 'ex3', 'file_link': 'link3'},
    ...                                {'extract': 'ex4', 'file_link': 'link4'}]}}

    >>> print_recomposed_dictionary(result_dict)
    Key: 1
    Question: Q1
    Answer: A1
    Sources:
      - Source 1:
        - Extract: ex1
        - File Link: link1
      - Source 2:
        - Extract: ex2
        - File Link: link2

    Key: 2
    Question: Q2
    Answer: A2
    Sources:
      - Source 1:
        - Extract: ex3
        - File Link: link3
      - Source 2:
        - Extract: ex4
        - File Link: link4

    """
    # Display the generated dictionary
    for key, result in in_dictionary.items():
        print(f"Key: {key}")
        print(f"Question: {result['question']}")
        print(f"Answer: {result['answer']}")
        print("Sources:")

        for source_number, source in enumerate(result['sources'], 1):
            print(f"  - Source {source_number}:")
            print(f"    - Extract: {source['extract']}")
            print(f"    - File Link: {source['file_link']}")

        print()



# --------------------------------
# [FUNCTIONS TO PROCESS QUESTIONS AND DERIVE ANSWERS]
# --------------------------------

def qa_chained_history(in__question: str,                 # Input question | #pylint: disable=W0102
                       in__llm: object,                   # Language model object
                       in__chat_history: list = [],       # List of chat history (default empty)
                       in__vectordb: Chroma = vectordb):  # VectorDB object (default is vectordb)

    """
    Perform conversational question-answering using a chained history approach.

    Parameters:
    in__question (str): The input question to be answered.
    in__llm (object): The language model used for answering questions.
    in__chat_history (list, optional): List of chat history messages (default is an empty list).
    in__vectordb (Chroma): VectorDB object for information retrieval (default is vectordb).

    Returns:
    dict: A JSON response object containing the question, answer, and source documents.

    """

    # Initialize a ConversationalRetrievalChain from the given language model and VectorDB object
    question_answer = ConversationalRetrievalChain.from_llm(
        llm=in__llm,                            # Language model                    # type: ignore
        retriever=in__vectordb.as_retriever(),  # Convert the VectorDB to a retriever
        verbose=True,                           # Enable verbose information
        return_source_documents=True            # Include source documents in the response
    )

    # Query the ConversationalRetrievalChain with the provided question and chat history
    result = question_answer({
        "question": in__question,               # Input question
        "chat_history": in__chat_history        # Chat history messages
    })

    return result  # Return the JSON response object


def process_question_chained_memorybuffer(in__question:str,
                                          in__llm: object,
                                          in__vectordb=vectordb):
    """
    Receive a question with used vectorstore and generate an answer including the document source

    Parameters
    ----------
    in_question : str
        question being raised
    in_vectordb : _type_
        vectorstore
    """

    # CONFIGURATION

    # Memory object, which is necessary to track the inputs/outputs and hold a conversation.
    memory = ConversationBufferMemory(memory_key="chat_history",
                                      return_messages=True,
                                      input_key="question",
                                      output_key="answer")


    question_answer = ConversationalRetrievalChain.from_llm(llm=in__llm,              # type: ignore
                                               retriever=in__vectordb.as_retriever(),
                                               verbose=True,
                                               memory=memory,
                                               return_source_documents=True)


    result = question_answer({"question": in__question})

    # Create the JSON response object
    total_response = {
        "question": result["question"],
        "result": {
            "answer": result["answer"],
            "source_documents": result["source_documents"]
        }
    }

    return total_response


if __name__ == "__main__":
    result_dict = {}
    logging.info("Start of script execution: %s", os.path.basename(__file__))

    llm_temp0 = OpenAI(temperature=0)       #type: ignore

    local_answer = get_local_answer(FILEPATH_DEBUGANSWER_PKL)

    question, answer, sources, history = qa_chained_history_decompose(local_answer)
    print(f"question: {question}\nanswer: {answer}\nsources: {sources}\nhistory: {history}")

    result_dict = recompose(in_dictionary=result_dict,
                            in_question=question,
                            in_answer=answer,
                            in_sources=sources)

    print_recomposed_dictionary(result_dict)


    logging.info("End of script: %s\n\n", os.path.basename(__file__))
