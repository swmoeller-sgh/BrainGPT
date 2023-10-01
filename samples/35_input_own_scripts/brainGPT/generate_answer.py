""" Module providing question - answer functions using an existing vectorstore"""

import os
import openai

from dotenv import load_dotenv, find_dotenv

from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory


from chromadb.config import Settings

# initialize the environment
env_file = find_dotenv(".env")
load_dotenv(env_file)


# LOAD ENVIRONMENT VARIABLES #

# openAI-settings
openai.api_key = os.getenv("OPENAI_API_KEY")

# directory definition
output_directory = os.environ.get("OUTPUT_DIRECTORY")
doc_source = os.environ.get("DOC_SOURCE_DIRECTORY")
PERSIST_DIRECTORY = str(os.environ.get('SIMPLEGPT_PERSIST_DIRECTORY'))

# LLM settings
embeddings_model_name = os.environ.get('EMBEDDINGS_MODEL_NAME')

# Define the Chroma settings
chroma_settings = Settings(
    chroma_db_impl='duckdb+parquet',
    persist_directory=PERSIST_DIRECTORY,
    anonymized_telemetry=False
    )


# CONFIGURATION
USED_MODEL = "OPENAI"

embeddings = OpenAIEmbeddings() # type: ignore
vectordb = Chroma(persist_directory=PERSIST_DIRECTORY,
                  embedding_function=embeddings)





# FUNCTIONS TO PROCESS QUESTIONS AND DERIVE ANSWERS #

def process_question_isolated(in_question:str,
                              in_llm: object,
                              in_vectordb=vectordb):
    """
    Receive a question with used vectorstore and generate an answer including the document source

    Parameters
    ----------
    in_question : str
        question being raised
    in_vectordb : _type_
        vectorstore
    """
    # Load the vectorstore
    # Creating a RetrievalQA object for question answering
    qa = RetrievalQA.from_chain_type(llm=in_llm,                            # type: ignore
                                     chain_type="stuff",
                                     retriever=in_vectordb.as_retriever(),
                                     return_source_documents=True)          # type: ignore

    # Get the result from the QA model using the input question
    result = qa({"question": in_question})  # Using the question to get an answer and associated source documents

    # Create the JSON response object
    total_response = {
        "question": result["question"], # The original question
        "result": {
            "answer": result["answer"], # The answer generated by the model
            "source_documents": result["source_documents"] # List of source documents relevant to the answer
        }
    }
    return total_response # List of source documents relevant to the answer


def process_question_chained(in_question:str,
                             in_llm: object,        # Default value for Language Model object
                             in_chat_history: list = [],
                             in_vectordb: Chroma =vectordb):    # Default value for VectorDB object
    """
    Receive a question, a context (chat history) with used vectorstore and generate an answer including the document source

    Parameters
    ----------
    in_question : str
        question being raised
    in_llm: object
        used language model
    in_chat_history: list of tuple
        previous questions and their answers
    in_vectordb : 
        vectorstore

    Returns
    -------
    json
        JSON object containing question and result including the answer and the source document
    """


    qa = ConversationalRetrievalChain.from_llm(llm=in_llm,    # Language Model object          # type: ignore
                                               retriever=in_vectordb.as_retriever(), # VectorDB object converted to retriever
                                               verbose=True,                    # Print verbose information
                                               return_source_documents=True)    # Include source documents in the result


    # Query the ConversationalRetrievalChain with the given question and chat history
    result = qa({"question": in_question,
                 "chat_history": in_chat_history})

    # Create the JSON response object
    total_response = {
        "question": result["question"],
        "result": {
            "answer": result["answer"],
            "source_documents": result["source_documents"]  # Include source documents in the response
        }
    }

    return total_response   # Return the JSON response object


def process_question_chained_memorybuffer(in_question:str,
                                          in_llm: object,
                                          in_vectordb=vectordb):
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


    qa = ConversationalRetrievalChain.from_llm(llm=in_llm,              # type: ignore
                                               retriever=in_vectordb.as_retriever(),
                                               verbose=True,
                                               memory=memory,
                                               return_source_documents=True)


    result = qa({"question": in_question})

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

    response = process_question_chained(in_question="How to describe VSM in three bullet points?",
                                        in_llm=llm_temp0,
                                        in_vectordb=vectordb)
    print(response)
