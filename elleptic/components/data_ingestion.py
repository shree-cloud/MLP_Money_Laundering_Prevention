from elleptic import utils
from elleptic.entity import config_entity
from elleptic.entity import artifact_entity
from elleptic.logger import logging
from elleptic.exception import EllepticException
import os,sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from elleptic import utils
from  elleptic.config import in_col


class DataIngestion:
    
    def __init__(self,data_ingestion_config:config_entity.DataIngestionConfig):
        try:
            logging.info(f"{'>>'*20} Data Ingestion {'<<'*20}")
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise EllepticException(e, sys)

    def initiate_data_ingestion(self)->artifact_entity.DataIngestionArtifact:
        try:
            logging.info("Exporting collection data as pandas dataframe")
            #exporting collection data as pandas dataframe
            df:pd.DataFrame = utils.get_collection_as_dataframe(
                database_name=self.data_ingestion_config.database_name,
                collection_name=self.data_ingestion_config.collection_name)
            exclude_columns=[in_col]
            df = utils.convert_column_to_required_dtype(df=df, exclude_columns=exclude_columns)

            #replace with Nan
            df.replace(to_replace="na", value=np.NAN, inplace=True)

            # df = df.drop(df[df['class']=='unknown'].index)


            #save data in feature store
            logging.info("Save data in feature store")
            #create feature store if not available
            logging.info("Create feature store if not available")
            feature_store_dir = os.path.dirname(self.data_ingestion_config.feature_store_file_path)
            os.makedirs(feature_store_dir, exist_ok=True)

            logging.info("Save df to feature store folder")
            #Save df to feature store folder
            df.to_csv(path_or_buf=self.data_ingestion_config.feature_store_file_path, index=False,header=True)

            logging.info("Split dataset into train and test set")
            #Split dataset into train and test set
            train_df, test_df = train_test_split(df,test_size=self.data_ingestion_config.test_size, random_state=42)

            logging.info("create dataset directory folder if not available")
            #create dataset directory folder if not available
            dataset_dir = os.path.dirname(self.data_ingestion_config.train_file_path)
            os.makedirs(dataset_dir,exist_ok=True)

            logging.info("Save df to feature store folder")
            train_df.to_csv(path_or_buf=self.data_ingestion_config.train_file_path, index=False, header=True)
            test_df.to_csv(path_or_buf=self.data_ingestion_config.test_file_path, index=False, header=True)

            #Prepare Artifact
            data_ingestion_artifact=artifact_entity.DataIngestionArtifact(
                feature_store_file_path=self.data_ingestion_config.feature_store_file_path,
                train_file_path=self.data_ingestion_config.train_file_path,
                test_file_path=self.data_ingestion_config.test_file_path
                )
            
            logging.info(f"Data Ingestion artifact: {data_ingestion_artifact}")
            return data_ingestion_artifact

        except Exception as e:
            raise EllepticException(e, sys)