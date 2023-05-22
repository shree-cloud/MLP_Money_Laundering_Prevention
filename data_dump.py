import pandas as pd
import pymongo
import json
import os

from elleptic.config import mongo_client

DATA_FILE_PATH = "/config/workspace/MLP_training_dataset.csv"
DATABASE_NAME = "mlp"
COLLECTION_NAME = "elleptic"

if __name__=="__main__":
    df = pd.read_csv(DATA_FILE_PATH)
    
    df.drop('Unnamed: 0', axis=1,inplace=True)
    print(f"rows and columns: {df.shape}")

    #convert dataframe to json to dump these records in mongodb
    df.reset_index(drop=True,inplace=True)

    json_record = list(json.loads(df.T.to_json()).values())
    print(json_record[0])

    mongo_client[DATABASE_NAME][COLLECTION_NAME].insert_many(json_record)