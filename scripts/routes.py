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
   answers when debugging is enabled. It also maintains a historic chat log of user questions and answers for reference.

"""
# [IMPORTS of modules and packages]
# Import system packages
import os
import logging

import forms

from flask import render_template
from langchain.llms import OpenAI


# Import own packages
#pylint: disable=E0401
from brain_gpt_config import config_braingpt
from braingpt_src import generate_answer

from braingpt_query import app

# [DIRECTORY setup]
# Define the root directory for file operations
ROOT_DIR = os.path.join(str(os.getenv("ROOT_DIR")))

# Define the path to the query log file
BRAINGPT_INQUIRY_LOG_FILE = os.path.join(ROOT_DIR, str(os.getenv("LOG_DIR")),
                                         str(os.getenv("QUERY_LOG_NAME")))

# [LOGGING settings]
config_braingpt.setup_logging(in__log_path_file= BRAINGPT_INQUIRY_LOG_FILE)
logging.info("Start of script execution: %s", os.path.basename(__file__))


# Define if local testing is done (i.e. without inquiring OpenAI)
DEBUG_MODE = os.getenv("DEBUG", "False") == "True"
print("DEBUG status:", DEBUG_MODE)

# Create LLM instances
llm_std = OpenAI()  # Standard OpenAI instance # type: ignore
llm_temp0 = OpenAI(temperature=0)  # OpenAI instance with temperature set to 0 # type: ignore

# Define the local testing debug_answer
debug_answer = {
    "question": "What is VSM?",
    "result": {
        "answer": "VSM stands for Value Stream Mapping. It is a process used to visually document and analyze the steps in a process to identify areas for improvement.",
        "source_documents": "Stefans own Source"
    }
}

# Store historic communication
historic_communication = []

# Define get_answer function
def get_answer(in__question, in__chat_history=[]):
    """
    Inquire the vector database on the question

    Parameters
    ----------
    in__question : str
        Input from form field

    Returns
    -------
    json
        Answer containing query, result, and document_references
    """
    answer = generate_answer.process_question_chained(in__question=in__question,
                                                      in__llm=llm_temp0,
                                                      in__chat_history=in__chat_history)
    return answer



# Definition of ROUTINGs
# Define the index route "/"
@app.route("/", methods=["GET", "POST"])
def index(in__debug_answer=debug_answer,
          in__debug_mode=DEBUG_MODE,
          in__historic_communication=historic_communication):
    """
    Definition of the index page

    Returns
    -------
    html generated from template
        Updated index page (from jinja template)
    """
    form = forms.RaiseQuestionForm()

    if form.validate_on_submit() is True:
        print("Continue conversation: ", form.new.data)

        if form.new.data is False:
            in__historic_communication = []

        question = form.question.data

        print("global variable:", DEBUG_MODE)
        if in__debug_mode is True:
            print("DEBUG mode activated, pre-configured answers only!")
            answer = in__debug_answer

        else:
            print("DEBUG mode deactivated!")
            # FIXME: Reference documents not shown
            answer = get_answer(question, in__chat_history=in__historic_communication)

        update_historic_chat = (question, answer["result"]["answer"])
        in__historic_communication.append(update_historic_chat)
        print(in__historic_communication)

        return render_template("index.html",
                               form=form,
                               question=question,
                               answer=answer,
                               historic_chat=in__historic_communication
                               )

    return render_template(template_name_or_list="index.html",
                           form=form)