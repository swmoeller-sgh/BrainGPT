"""
[Purpose]
=========
Query our own data

[References]
============

"""

# // TODO: Move to local processing for better efficiency (see LinkedIn article and Obsidian notes).
# // TODO: Implement an empty form at the beginning for user input.
# // FIXME: Address the issue of an empty historic chat at the beginning.

# [IMPORTS of modules and packages]
import logging
import os  # Operating System module for interacting with the OS environment

from dotenv import load_dotenv, find_dotenv # Dotenv library for loading env. variables from a file

import openai  # OpenAI library for working with OpenAI's GPT-3 or other models

from flask import Flask

from braingpt_config import config_braingpt #pylint: disable=E0401


# [INITIALIZE environment]
env_file = find_dotenv(".env")
load_dotenv(env_file)

# [IMPORTS of environment constants]
openai.api_key = os.getenv("OPENAI_API_KEY")        # Set the OpenAI API key from env. variables

# [DIRECTORY setup]
ROOT_DIR = os.path.join(str(os.getenv("ROOT_DIR"))) # Define the root directory for file operations
BRAINGPT_INQUIRY_LOG_FILE = os.path.join(ROOT_DIR,  # Define the path to the query log file
                                         str(os.getenv("LOG_DIR")),
                                         str(os.getenv("QUERY_LOG_NAME")))

# [LOGGING settings]
config_braingpt.setup_logging(in__log_path_file= BRAINGPT_INQUIRY_LOG_FILE)
logging.info("Start of script execution: %s", os.path.basename(__file__))

# [TESTING retrival from environment, i.e. local testing without inquiring OpenAI]
DEBUG_MODE = os.getenv("DEBUG", "False") == "True"
logging.info("DEBUG mode: %s", DEBUG_MODE)

# [CONFIGURATION of flask-application]
app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY")
from routes import *    # routes to pages #pylint: disable=E0401, C0411, C0413, W0401


# [MAIN part]
if __name__ == "__main__":
    # Start the Flask application on host 127.0.0.1 and port 4455 for debugging
    app.run(host="127.0.0.1",  # type: ignore
            port=4455,
            debug=True)
    
