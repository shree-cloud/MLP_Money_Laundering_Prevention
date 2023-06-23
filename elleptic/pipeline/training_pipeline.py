from elleptic.logger import logging
from elleptic.exception import EllepticException
from elleptic.utils import get_collection_as_dataframe
import os,sys
from elleptic.entity import config_entity
from elleptic.components.data_ingestion import DataIngestion
from elleptic.components.data_validation import DataValidation
from elleptic.components.data_transformation import DataTransformation
from elleptic.components.model_trainer import ModelTrainer
from elleptic.components.model_evaluation import ModelEvaluation
from elleptic.components.model_pusher import ModelPusher
import streamlit as st


def initiate_training_pipeline():
    try:
        training_pipeline_config = config_entity.TrainingPipelineConfig()


        st.markdown("Data Ingestion")
        data_ingestion_config  = config_entity.DataIngestionConfig(training_pipeline_config=training_pipeline_config)
        print(data_ingestion_config.to_dict())
        data_ingestion = DataIngestion(data_ingestion_config=data_ingestion_config)
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()

        st.markdown("Validating the Data")
        data_validation_config=config_entity.DataValidationConfig(training_pipeline_config=training_pipeline_config)
        data_validation = DataValidation(data_validation_config=data_validation_config, data_ingestion_artifact=data_ingestion_artifact)
        data_validation_artifact = data_validation.initiate_data_validation()

        #Data Transformation
        st.markdown("Transforming the Data")
        data_transformation_config = config_entity.DataTransformationConfig(training_pipeline_config=training_pipeline_config)
        data_transformation = DataTransformation(data_transformation_config=data_transformation_config, data_ingestion_artifact=data_ingestion_artifact)
        data_transformation_artifact = data_transformation.initiate_data_transformation()

        #Model Trainer
        st.markdown("Training the Model")
        model_trainer_config = config_entity.ModelTrainerConfig(training_pipeline_config=training_pipeline_config)
        model_trainer = ModelTrainer(model_trainer_config=model_trainer_config, data_transformation_artifact= data_transformation_artifact)
        model_trainer_artifact = model_trainer.initiate_model_trainer()

        #model Evaluation
        st.markdown("Evaluating the Model")
        model_evaluation_config = config_entity.ModelEvaluationConfig(training_pipeline_config=training_pipeline_config)
        model_evaluation = ModelEvaluation(model_evaluation_config = model_evaluation_config,
            data_ingestion_artifact= data_ingestion_artifact,
            data_transformation_artifact=data_transformation_artifact, 
            model_trainer_artifact=model_trainer_artifact)
        model_evaluation_artifact = model_evaluation.initiate_model_evaluation()

        # Model Pusher
        st.markdown("Pushing the model to the for Prediction pipeline")
        model_pusher_config = config_entity.ModelPusherConfig(training_pipeline_config=training_pipeline_config)
        model_pusher = ModelPusher(model_pusher_config=model_pusher_config,
        data_transformation_artifact=data_transformation_artifact,
        model_trainer_artifact=model_trainer_artifact)
        model_pusher_artifact = model_pusher.initiate_model_pusher()
    
    except Exception as e:
          raise EllepticException(e, sys)