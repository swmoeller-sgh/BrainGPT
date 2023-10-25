""" 
Purpose
1. Web Application for Querying Information: 
   The script serves as the backend of a web application designed for querying information. 
   It uses the Flask framework to create routes and handle HTTP requests.

2. Interaction with OpenAI GPT-3: It interacts with OpenAI's GPT-3 model to generate 
   answers to user questions. The `get_answer` function sends user questions to the model and 
   processes the responses.

3. Debugging and Testing: The script includes debugging features controlled by 
   the `DEBUG_MODE` variable, allowing developers to test the application with pre-configured 
   answers when debugging is enabled. It also maintains a historic chat log of user questions 
   and answers for reference.

"""
# [IMPORTS of modules and packages]
# Import system packages
import os
import logging
import uuid
# from datetime import datetime

# import forms    #pylint: disable=E0401

import json

from flask import render_template, request, session
from langchain.llms.openai import OpenAI


# Import own packages
from braingpt_app import app
from braingpt_config import config_braingpt     #pylint: disable=E0401
from braingpt_src import generate_answer        #pylint: disable=E0401




# [CONSTANTS and CONFIGURATIONS]
# Directory setup
ROOT_DIR = os.path.join(str(os.getenv("ROOT_DIR"))) # Define the root directory for file operations
BRAINGPT_INQUIRY_LOG_FILE = os.path.join(ROOT_DIR,  # Define the path to the query log file
                                         str(os.getenv("LOG_DIR")),
                                         str(os.getenv("QUERY_LOG_NAME")))
question_chain=[]
# historic_communication = []             # Store historic communication
chat_history = []


# [LOGGING settings]
config_braingpt.setup_logging(in__log_path_file= BRAINGPT_INQUIRY_LOG_FILE)
logging.info("\n\n")
logging.info("Start of script execution: %s", os.path.basename(__file__))


# [TESTING retrival from environment, i.e. local testing without inquiring OpenAI]
DEBUG_MODE = os.getenv("DEBUG", "False") == "True"
logging.info("DEBUG status: %s\n", DEBUG_MODE)


# [LLM INSTANCES creation]
llm_std = OpenAI()                  # Standard OpenAI instance # type: ignore
llm_temp0 = OpenAI(temperature=0)   # OpenAI instance with temperature set to 0 # type: ignore


# [FUNCTION definition]
def get_answer(in__question,            # Define get_answer function        #pylint: disable=W0102
               in__chat_history,
               in__llm):
    """
    Query information from the language model.

    Parameters
    ----------
    in__question : str
        Input question from the form field.
    in__chat_history : list, optional
        Chat history for maintaining conversation context.

    Returns
    -------
    dict
        JSON response containing query, result, and document_references.
    """

    result = generate_answer.qa_chained_history(in__question=in__question,
                                                      in__llm=in__llm,
                                                      in__chat_history=in__chat_history)
    return result

def get_local_answer(in__filepath):
    """
    Read a local answer from a text file.

    Parameters
    ----------
    in__filepath : str
        File path to the local answer.

    Returns
    -------
    str
        Content of the local answer as a string.
    """

    try:
        with open(in__filepath, "r", encoding="utf-8") as file:
            sample_result = file.read()
    except FileNotFoundError:
        print(f"File '{in__filepath}' not found.")
        sample_result = None

    return sample_result


def decompose_answer(in__json):

    input_question = in__json['question']
    output_answer = in__json['result']['answer']

    # Extract and shorten the page content for each source document to a maximum of 50 letters
    sources = []
    for doc in in__json['result']['source_documents']:
        short_content = doc['page_content'][0:50]
        source_url = doc['metadata']['source']
        sources.append([short_content, source_url])

    return input_question, output_answer,sources


def structure_answer(in__json):
    """
    Create a nested dictionary from a JSON input, where each question is associated with an answer 
    and potential source information.

    Parameters:
    in__json (list): A list of question-answer pairs in JSON format.

    Returns:
    dict: A nested dictionary where each unique identifier (UUID) is associated with a question 
          and answer data structure. 
          If the answer contains phrases like "I don't know," the source information is left as 
          a placeholder.
    """

    # Reverse the chat history to have the most recent entry at the top
    in__json = in__json[::-1]

    # Initialize the nested dictionary
    nested_dict = {}

    for question, answer in in__json:
        # Clean up the answer by removing leading and trailing whitespaces and '\n'
        answer = answer.strip()

        # Check if the answer contains "I don't know" or similar phrases to classify it as a source
        if "I don't know" in answer:
            answer_data = {
                'Answer': None,
                'Source': None  # Placeholder for source (to be added later)
            }
        else:
            answer_data = {
                'Answer': answer,
                'Source': None  # Placeholder for source (to be added later)
            }

        # Generate a unique UUID based on the question, date, and time
        unique_id = uuid.uuid5(uuid.NAMESPACE_DNS, f"{question}_{str(uuid.uuid1())}")

        # Create or update the nested dictionary
        nested_dict[str(unique_id)] = {question: answer_data}

    return nested_dict


debug_answer = get_local_answer(in__filepath="data/sample_answer05.txt")


# [ROUTING definition]
@app.route("/", methods=["GET"])         # Define the index route "/"
def index():                             #pylint: disable=W0102
    """
    Define the index route ("/").

    Returns
    -------
    str
        Rendered template for the index page.
    """

    return render_template("index.html")


@app.route("/question", methods=["GET", "POST"])        # Define the question route "/question"
def question(in__llm=llm_temp0,
             in__debug_mode=DEBUG_MODE,
             in__debug_answer=debug_answer,              #pylint: disable=W0102
             ):
    """
    Define the question route ("/question").

    Returns
    -------
    str
        Rendered template for the question page with the answer.
    """

    form_question = ""
    proc_question = ""
    answer = ""
    source = ""
    format_chat_history = ""

# // TODO Optimize ChromaDB to utilize different collections
# // TODO Include history / reference to source
# // TODO Investigate the application of different languages



    # Initialize chat history from the session or create an empty list
    chat_history = session.get('chat_history', [])

    if request.method == "POST":

        continue_conversation = request.form.get("c_conv")
        logging.info("Contimue conversation: %s", continue_conversation)

        collection_business = request.form.get("business")
        if collection_business == "True":
            logging.info("Result of business checkbox: %s", collection_business)

        form_question = request.form["form_question"]
        logging.info("Question stated: %s\n",form_question)

        if in__debug_mode is True:
            out_answer_dict = in__debug_answer
#            proc_question, answer, source = decompose_answer(in__json=out_answer_dict)
            proc_question = out_answer_dict["question"]
            answer = out_answer_dict["result"]["answer"]
            source = out_answer_dict["result"]["source_documents"]

        else:
            if continue_conversation != "True":
                chat_history=[]

            out_answer_dict = get_answer(in__question=form_question,
                                    in__chat_history=chat_history,
                                    in__llm=in__llm)
#            proc_question, answer, source = decompose_answer(in__json=out_answer_dict)

            proc_question = out_answer_dict["question"]
            answer = out_answer_dict["result"]["answer"]
            source = out_answer_dict["result"]["source_documents"][:20]
            chat_history.append((proc_question,answer))
            logging.info("Chat history: %s\n",chat_history)


        # Store the updated chat history in the session
        session['chat_history'] = chat_history
        format_chat_history = structure_answer(chat_history)

    return render_template("question.html",
                           form_question=form_question,
                           question = proc_question,
                           answer = answer,
                           source = source,
                           history = format_chat_history)
