""" Module providing question - answer functions using an existing vectorstore"""

# [IMPORT of modules and packages]
import os
import logging

import openai

from dotenv import load_dotenv, find_dotenv

# // TODO Neuinstallation env: langchain.vectorstores --> langchain.vectorstores.chroma wechseln
from langchain.llms.openai import OpenAI
from langchain.vectorstores.chroma import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory

from chromadb.config import Settings

#pylint: disable=E0401

from braingpt_config import config_braingpt # type: ignore


# [INITIALIZE environment]
env_file = find_dotenv(".env")
load_dotenv(env_file)

# [DIRECTORY definition]
ROOT_DIR = os.path.join(str(os.getenv("ROOT_DIR")))
BRAINGPT_GENERATE_ANSWER_LOG_FILE = os.path.join(ROOT_DIR, str(os.getenv("LOG_DIR")),
                                     str(os.getenv("QUERY_LOG_NAME")))


config_braingpt.setup_logging(in__log_path_file=BRAINGPT_GENERATE_ANSWER_LOG_FILE)

# [LOAD enironment variables]
openai.api_key = os.getenv("OPENAI_API_KEY")        # openAI-settings

#output_directory = os.environ.get("OUTPUT_DIRECTORY")
#doc_source = os.environ.get("DOC_SOURCE_DIRECTORY")

PROCESSED_DATA_DIR = os.path.join(ROOT_DIR,
                                  str(os.getenv("PROCESSED_DATA_DIR")),
                                  str(os.getenv("CHROMA_DB")))
logging.info("Target directory for processed files: %s", PROCESSED_DATA_DIR)

# PERSIST_DIRECTORY = str(os.environ.get('SIMPLEGPT_PERSIST_DIRECTORY'))

# LLM settings
#embeddings_model_name = os.environ.get('EMBEDDINGS_MODEL_NAME')

# Define the Chroma settings
CHROMA_SETTINGS = Settings(
        persist_directory=PROCESSED_DATA_DIR,
        anonymized_telemetry=False
)


# CONFIGURATION
USED_MODEL = "OPENAI"

embeddings = OpenAIEmbeddings() # type: ignore
vectordb = Chroma(persist_directory=PROCESSED_DATA_DIR,
                  embedding_function=embeddings)





# FUNCTIONS TO PROCESS QUESTIONS AND DERIVE ANSWERS #

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

    # Create a JSON response object with the question, answer, and source documents
    total_response = {
        "question": result["question"],
        "result": {
            "answer": result["answer"],
            "source_documents": result["source_documents"]  # Include source docs in the response
        }
    }

    return total_response  # Return the JSON response object


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
    llm_temp0 = OpenAI(temperature=0)       #type: ignore

    response = qa_chained_history(in__question="How to describe VSM in three bullet points?",
                                        in__llm=llm_temp0,
                                        in__vectordb=vectordb)
    print(response)
