# config.py
import os
import logging
from dotenv import load_dotenv, find_dotenv

# [INITIALIZE environment]
env_file = find_dotenv(".env")
load_dotenv(env_file)

"""
class Config:
    SECRET_KEY = os.getenv("FLASK_SECRET_KEY")
    # Other configuration settings...

class DevelopmentConfig(Config):
    DEBUG = True

class ProductionConfig(Config):
    DEBUG = False
"""

# [LOGGING settings]
def setup_logging(in__log_path_file:str):
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s : %(levelname)s %(filename)s : Line No. : %(lineno)d - %(message)s",
                        filename=in__log_path_file,
                        filemode='a')
