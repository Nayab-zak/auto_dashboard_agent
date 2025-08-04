# config.py

from dotenv import load_dotenv
load_dotenv()
import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
VERTICA_HOST = os.getenv("VERTICA_HOST")
VERTICA_PORT = os.getenv("VERTICA_PORT")
VERTICA_DB = os.getenv("VERTICA_DB")
VERTICA_USER = os.getenv("VERTICA_USER")
VERTICA_PASSWORD = os.getenv("VERTICA_PASSWORD")

VERTICA_CONFIG = {
    "server": VERTICA_HOST,
    "database": VERTICA_DB,
    "uid": VERTICA_USER,
    "pwd": VERTICA_PASSWORD
}

VERTICA_TABLE_VOYAGES = os.getenv("VERTICA_TABLE_VOYAGES")
VERTICA_TABLE_LINES = os.getenv("VERTICA_TABLE_LINES")
VERTICA_TABLE_BOX_STAT = os.getenv("VERTICA_TABLE_BOX_STAT")
