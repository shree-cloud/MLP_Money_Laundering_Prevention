import pymongo
import pandas as pd
import json
from dataclasses import dataclass
import os
import streamlit as st

# Provide the mongodb localhost url to connect python to mongodb.

@dataclass
class EnvironmentVariable:
    # mongo_db_url:str = os.getenv("MONGO_DB_URL")
    mongo_db_url:str = st.secrets["MONGO_DB_URL"]


env_var = EnvironmentVariable()
mongo_client = pymongo.MongoClient(env_var.mongo_db_url)
TARGET_COLUMN = ["class"]
in_col = ["class","txId","time step"]
