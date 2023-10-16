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
from langchain.llms import OpenAI


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
               in__chat_history):
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

    answer = generate_answer.process_question_chained(in__question=in__question,
                                                      in__llm=llm_temp0,
                                                      in__chat_history=in__chat_history)
    return answer


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
            sample_answer = file.read()
    except FileNotFoundError:
        print(f"File '{in__filepath}' not found.")
        sample_answer = None

    return sample_answer


def structure_answer(in__json):

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



# Derive the local testing debug_answer
local_answer = get_local_answer(in__filepath="data/sample_answer02.txt")
debug_answer = json.loads(local_answer) # type: ignore
logging.info("Local answer loaded.")



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
def question(in__debug_mode=DEBUG_MODE,
             in__debug_answer=debug_answer              #pylint: disable=W0102
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
            proc_question = out_answer_dict["question"]
            answer = out_answer_dict["result"]["answer"]
            source = out_answer_dict["result"]["source_documents"]

        else:
            if continue_conversation != "True":
                chat_history=[]
        
            out_answer_dict = get_answer(in__question=form_question, 
                                    in__chat_history=chat_history)
            proc_question = out_answer_dict["question"]
            answer = out_answer_dict["result"]["answer"]
            source = out_answer_dict["result"]["source_documents"][:20]
            chat_history.append((proc_question,answer))
            logging.info("Chat history: %s\n",chat_history)
     
            
        # Store the updated chat history in the session
        session['chat_history'] = chat_history
        format_chat_history = structure_answer(chat_history)

#      document_names = [document.get('metadata', {}).get('source', '')[:20] for document in source]


        #date_time = str(datetime.now())

#        question_chain.append({"date": date_time, "question": in_question})
#        print(question_chain)
#        logging.info(question_chain)

    return render_template("question.html",
                           form_question=form_question,
                           question = proc_question,
                           answer = answer,
                           source = source,
                           history = format_chat_history)
