import pandas as pd
import numpy as np
from elleptic.logger import logging
from elleptic.exception import EllepticException
from elleptic.config import mongo_client
import os,sys
import yaml


def get_collection_as_dataframe(database_name:str,collection_name:str)->pd.DataFrame:
    try:
        logging.info(f"Reading data from database: {database_name} and collection: {collection_name}")
        df = pd.DataFrame(list(mongo_client[database_name][collection_name].find()))
        logging.info(f"Found columns: {df.columns}")
        if "_id" in df.columns:
            logging.info(f"Dropping column: _id")
            df = df.drop("_id",axis=1)
        logging.info(f"Row and Columns in df: {df.shape}")
        return df
    except Exception as e:
        raise EllepticException(e, sys)

def write_yaml_file(file_path, data:dict):
    try:
        file_dir = os.path.dirname(file_path)
        os.makedirs(file_dir, exist_ok=True)
        with open(file_path,"w") as file_writer:
            yaml.dump(data,file_writer)
    except Exception as e:
        raise EllepticException(e, sys)

def convert_column_to_required_dtype(df:pd.DataFrame,exclude_columns:list)->pd.DataFrame:
    try:
        # for column in df.columns:
            # if column not in exclude_columns:
            #     df[column] = df[column].astype('float')
        for ex_col in exclude_columns:
            df[ex_col] = df[ex_col].astype(int)
        return df
    except Exception as e:
        raise EllepticException(e, sys)