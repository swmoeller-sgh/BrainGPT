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
# --------------------------------
# [IMPORTS of modules and packages]
# --------------------------------
# Import system packages
import os
import logging

from flask import render_template, request, session
from langchain.llms.openai import OpenAI

# Import own packages
from braingpt_app import app
from braingpt_config import config_braingpt     #pylint: disable=E0401
from braingpt_src import generate_answer        #pylint: disable=E0401


# --------------------------------
# [DIRECTORY setup]
# --------------------------------
ROOT_DIR = os.path.join(str(os.getenv("ROOT_DIR"))) # Define the root directory for file operations
BRAINGPT_INQUIRY_LOG_FILE = os.path.join(ROOT_DIR,  # Define the path to the query log file
                                         str(os.getenv("LOG_DIR")),
                                         str(os.getenv("QUERY_LOG_NAME")))
FILEPATH_DEBUGANSWER_PKL = "data/10_raw/qa_chained_history_sample_answer.pkl"


# --------------------------------
# [LOGGING initial state]
# --------------------------------
config_braingpt.setup_logging(in__log_path_file= BRAINGPT_INQUIRY_LOG_FILE)
logging.info("\n\n")
logging.info("Start of script execution: %s", os.path.basename(__file__))


# --------------------------------
# [DEFINITION OF CONSTANTS]
# --------------------------------




# --------------------------------
# [SETUP WORKING MODE (DEBUG y/n)]
# --------------------------------
DEBUG_MODE = os.getenv("DEBUG", "False") == "True"
logging.info("DEBUG status: %s\n", DEBUG_MODE)

# --------------------------------
# [CONFIGURATION]
# --------------------------------
llm_std = OpenAI()                  # Standard OpenAI instance # type: ignore
llm_temp0 = OpenAI(temperature=0)   # OpenAI instance with temperature set to 0 # type: ignore


# --------------------------------
# [SUPPORTING FUNCTIONS] to load debug answers, structure/decompose answers, etc.
# --------------------------------


# --------------------------------
# [ROUTING definition]
# --------------------------------
@app.route("/", methods=["GET"])         # Define the index route "/"
def index():                             #pylint: disable=W0102
    """
    Define the index route ("/").

    Returns
    -------
    str
        Rendered template for the index page.
    """
    session.clear()
    return render_template("index.html")


@app.route("/question", methods=["GET", "POST"])        # Define the question route "/question"
def question(in__llm=llm_temp0,
             in__debug_mode=DEBUG_MODE,
             in__debug_answer=FILEPATH_DEBUGANSWER_PKL
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
    reference = ""
    result = {}

# // TODO Optimize ChromaDB to utilize different collections



    # Initialize chat history from the session or create an empty list / dictionary
    chat_log = session.get("chat_log", {})
    chat_history = session.get('chat_history', [])

    if request.method == "POST":
        flag_continue_conversation = request.form.get("c_conv")
        logging.info("Contimue conversation: %s", flag_continue_conversation)

        flag_collection_business = request.form.get("business")
        if flag_collection_business == "True":
            logging.info("Result of business checkbox: %s", flag_collection_business)

        form_question = request.form["form_question"]
        logging.info("Question stated: %s\n",form_question)

        if in__debug_mode is True:

            if flag_continue_conversation != "True":
                chat_history=[]
                chat_log={}
                session.clear()

            # generate answer
            result = generate_answer.get_local_answer(in__debug_answer)

            # extract single elements from the answer
            proc_question, answer, reference, history = \
                generate_answer.qa_chained_history_decompose(result)        #pylint: disable=W0612
            logging.info("Local answer loaded.")

            # add new question, answer and sources to chat_log
            chat_log= generate_answer.recompose(in_dictionary=chat_log,
                                                in_question=proc_question,
                                                in_answer=answer,
                                                in_sources=reference)
            chat_history= generate_answer.extract_qa_pairs(chat_log)

        else:
            if flag_continue_conversation != "True":
                chat_history=[]
                chat_log={}
                session.clear()
            result= generate_answer.qa_chained_history(in__question=form_question,
                                                       in__chat_history=chat_history,
                                                       in__llm=in__llm)
            proc_question, answer, reference, history = \
                generate_answer.qa_chained_history_decompose(result)
            logging.info("Online answer loaded.")
            chat_log= generate_answer.recompose(in_dictionary=chat_log,
                                                in_question=proc_question,
                                                in_answer=answer,
                                                in_sources=reference)

            chat_history= generate_answer.extract_qa_pairs(chat_log)

            logging.info("Chat history: %s\n",chat_history)

        # Store the updated chat history in the session
        session['chat_history'] = chat_history
        session['chat_log']= chat_log


    return render_template("question.html",
                           form_question=form_question,
                           question = proc_question,
                           answer = answer,
                           source = reference,
                           history = chat_log)
