"GPT over our private data"
import os

from flask import Flask
from dotenv import load_dotenv


load_dotenv()

# // TODO - switch to local processing (see article in LinkedIn and in Obsedian)
# // TODO - empty form in the beginning
# // FIXME - empty historic chat in the beginning


# Define, if local testing is done (without inquiring OpenAI)
DEBUG_MODE = os.getenv("DEBUG", "False") == "True"

# Definition of GPT-relevant inputs
app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("FLASK_SECRET_KEY")

from routes import *

# main part
if __name__ == "__main__":
    app.run(host="127.0.0.1", # type: ignore
            port=4455,
            debug=True)
    