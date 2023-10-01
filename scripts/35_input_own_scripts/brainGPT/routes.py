"Routing file"

# import system packages
import os
import forms
from flask import render_template
from langchain.llms import OpenAI

# import own packages
from brain_app import app
from gpt_tools import generate_answer

# Define, if local testing is done (i.e. without inquiring OpenAI)
DEBUG_MODE = os.getenv("DEBUG", "False") == "True"
print("DEBUG status:", DEBUG_MODE)

# create LLM instances
llm_std = OpenAI()  # type: ignore
llm_temp0 = OpenAI(temperature=0)       #type: ignore

# Define the local testing debug_answer
debug_answer= {
    "question": "What is VSM?",
    "result": {
        "answer": "VSM stands for Value Stream Mapping. It is a process used to visually document and analyze the steps in a process to identify areas for improvement.",
        "source_documents": "Stefans own Source"
        }
}

historic_communication = []


# define get_answer function
def get_answer(in_question, 
               in_chat_history = []):
    """
    Inquiring the vector database on the question

    Parameters
    ----------
    question : str
        Input from form field

    Returns
    -------
    json
        Answer containing query, result and document_references
    """
    answer = generate_answer.process_question_chained(in_question = in_question,
                                                      in_llm=llm_temp0,
                                                      in_chat_history=in_chat_history)
    return answer


# Definition of ROUTINGs

# index / landing page "/"
@app.route("/", methods = ["GET", "POST"])
def index(in_debug_answer = debug_answer,
          in_DEBUG_MODE = DEBUG_MODE,
          in_historic_communication = historic_communication):
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
            in_historic_communication = []

        question = form.question.data

        print("global variable:", DEBUG_MODE)
        if DEBUG_MODE is True:
            print("DEBUG mode activated, pre-configured answers only!")
            answer = in_debug_answer

        else:
            print("DEBUG mode deactivated!")
# // FIXME - reference documents not shown
            answer = get_answer(question,
                                in_chat_history=in_historic_communication)

        update_historic_chat = (question, answer["result"]["answer"])
        in_historic_communication.append(update_historic_chat)
        print(in_historic_communication)

        return render_template("index.html", 
                               form = form, 
                               question = question, 
                               answer = answer, 
                               historic_chat = in_historic_communication
                               )

    return render_template("index.html", form = form)
