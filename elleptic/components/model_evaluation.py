import os,sys
from elleptic.entity import config_entity, artifact_entity
from elleptic.predictor import ModelResolver
from elleptic.exception import EllepticException
from elleptic.logger import logging
from elleptic.utils import load_object
from sklearn.metrics import f1_score
import pandas as pd
from elleptic.config import TARGET_COLUMN


class ModelEvaluation:
    def __init__(self,
                model_evaluation_config:config_entity.ModelEvaluationConfig,
                data_ingestion_artifact:artifact_entity.DataIngestionArtifact,
                data_transformation_artifact:artifact_entity.DataTransformationArtifact,
                model_trainer_artifact:artifact_entity.ModelTrainerArtifact
                ):
                try:
                    logging.info(f"{'>>'*20} Model Evaluation {'<<'*20}")
                    self.model_evaluation_config = model_evaluation_config
                    self.data_ingestion_artifact = data_ingestion_artifact
                    self.data_transformation_artifact = data_transformation_artifact
                    self.model_trainer_artifact = model_trainer_artifact
                    self.model_resolver = ModelResolver()
                except Exception as e:
                    raise EllepticException(e, sys)


    def initiate_model_evaluation(self,)->artifact_entity.ModelEvaluationArtifact:
        try:
            # if saved model folder has model then we will compare 
            # which model is best, trained or the model form saved model folder

            logging.info(f"if saved model folder has model then we will compare which model is best, trained or the model in saved_model folder")
            latest_dir_path = self.model_resolver.get_latest_dir_path()
            if latest_dir_path == None:
                model_evaluation_artifact = artifact_entity.ModelEvaluationArtifact(is_model_accepted=True,
                improved_accuracy=None)
                logging.info(f"Model Evaluation Artifact: {model_evaluation_artifact}")
                return model_evaluation_artifact

            # finding location of transformer, model and target encoder.
            logging.info("finding location of transformer, model and target encoder")
            transformer_path = self.model_resolver.get_latest_transformer_path()
            model_path = self.model_resolver.get_latest_model_path()
            target_encoder_path = self.model_resolver.get_latest_target_encoder_path()
            
            logging.info("loading previously trained objects")
            # loading previously trained objects
            transformer = load_object(file_path=transformer_path)
            model = load_object(file_path=model_path)
            target_encoder = load_object(file_path=target_encoder_path)

            logging.info("objects of currently trained model")
            # objects of currently trained model
            current_transformer = load_object(file_path=self.data_transformation_artifact.transform_object_path)
            current_model = load_object(file_path=self.model_trainer_artifact.model_path)
            current_target_encoder = load_object(file_path=self.data_transformation_artifact.target_encoder_path)

            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            target_df = test_df[TARGET_COLUMN]
            y_true = target_encoder.transform(target_df)

            # accuracy using previous trained model
            input_feature_name = list(transformer.feature_names_in_)
            input_arr = transformer.transform(test_df[input_feature_name])
            y_pred = model.predict(input_arr)
            # print(f"Prediction using previous model: {target_encoder.inverse_transform(y_pred[:5])}")
            previous_model_score = f1_score(y_true=y_true, y_pred=y_pred)

            logging.info(f"accuracy using previously trained model :{previous_model_score}")
            #accuracy using current trained model
            input_feature_name = list(current_transformer.feature_names_in_)
            input_arr = current_transformer.transform(test_df[input_feature_name])
            y_pred = current_model.predict(input_arr)
            y_true = current_target_encoder.transform(target_df)
            # print(f"Prediction using trained model: {current_target_encoder.inverse_transform(y_pred[:5])}")
            current_model_score = f1_score(y_true=y_true, y_pred=y_pred)
            logging.info(f"accuracy using currently trained model: {current_model_score}")

            if current_model_score < previous_model_score:
                logging.info(f"current trained model is not better than previous model")
                raise Exception("current trained model is not better than previous model")
            
            model_evaluation_artifact = artifact_entity.ModelEvaluationArtifact(is_model_accepted=True,
            improved_accuracy=current_model_score - previous_model_score)

            logging.info(f"Model Evaluation artifact: {model_evaluation_artifact}")
            return model_evaluation_artifact

        except Exception as e:
            raise EllepticException(e, sys)