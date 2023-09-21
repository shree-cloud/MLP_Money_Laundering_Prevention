import os,sys
from elleptic.entity import config_entity,artifact_entity
from elleptic.exception import EllepticException
from elleptic.logger import logging

import numpy as np
import pandas as pd
from elleptic import utils
from sklearn.ensemble import RandomForestClassifier
import streamlit as st
from sklearn.metrics import f1_score, classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV, KFold
from elleptic.config import in_col


class ModelTrainer:
    
    def __init__(self,model_trainer_config:config_entity.ModelTrainerConfig,
                data_transformation_artifact:artifact_entity.DataTransformationArtifact
                ):
                try:
                    logging.info(f"{'>>'*20} Model Trainer {'<<'*20}")
                    self.model_trainer_config = model_trainer_config
                    self.data_transformation_artifact = data_transformation_artifact
                except Exception as e:
                    raise EllepticException(e, sys)


    def fine_tune(self,x,y):
        try:
            logging.info(f"Initiating hyperparameter tuning")
            param_grid = {
                'class_weight': ["balanced", "balanced_subsample", {0: 0.1, 1: 1}, {0: 0.3, 1: 1}],
                'n_estimators': [50, 100, 200, 300],
                # 'max_features': ['sqrt', 'log2', 0.2, 0.5],
                'max_depth': [2, 8, 16, 30],
                'min_samples_split': [2, 4, 8, 10],
                'min_samples_leaf': [1, 2, 4]
                # 'bootstrap': [True, False]
            }

            # using RandomForestClassifier model
            kfold_validation = KFold(3)
            rf_model = RandomForestClassifier(random_state=42)

            # creating a GridSearchCV object and fit to data
            grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=kfold_validation, verbose = 2, n_jobs=4)
            logging.info(f"Fitting grid search CV to find best parameters")
            grid_search.fit(x,y)

            # best hyperparameters and score
            best_params = grid_search.best_params_
            best_score = grid_search.best_score_
            logging.info(f"Best parameters achieved after hyperparameter tuning = {best_params}")
            logging.info(f"Best accuracy score achieved after hyperparameter tuning = {best_score}")
            
            return best_params
        except Exception as e:
            raise EllepticException(e, sys)

    def train_model(self,x,y):
        try:
            # best_params = ModelTrainer.fine_tune(self, x, y)

            logging.info(f"Fitting the model with best parameters obtained after hyperparameter tuning")
            # clf = RandomForestClassifier(n_estimators=best_params['n_estimators'], class_weight=best_params['class_weight'], 
            #                             max_depth=best_params['max_depth'], min_samples_split=best_params['min_samples_split'], 
            #                             min_samples_leaf=best_params['min_samples_leaf'],
            #                             )


            clf = RandomForestClassifier(class_weight= 'balanced_subsample', max_depth= 16, min_samples_leaf= 1, min_samples_split= 2, n_estimators= 300)
        


            clf.fit(x,y)
            return clf
        except Exception as e:
            raise EllepticException(e, sys)

    def initiate_model_trainer(self,)->artifact_entity.ModelTrainerArtifact:
        try:
            logging.info(f"loading train and test array")
            train_arr = utils.load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_train_path)
            test_arr = utils.load_numpy_array_data(file_path=self.data_transformation_artifact.transformed_test_path)

            logging.info(f"Splitting input and target feature from both train and test arr")
            x_train, y_train =train_arr[:,:-1], train_arr[:,-1]
            x_test, y_test = test_arr[:,:-1], test_arr[:,-1]

            logging.info(f"Training Model")
            model = self.train_model(x=x_train,y=y_train)
            st.header("Model Trained Successfully")

            logging.info(f"Calculating f1 train score")
            yhat_train = model.predict(x_train)
            f1_train_score = f1_score(y_true=y_train,y_pred=yhat_train)
            training_accuracy = accuracy_score(y_train, yhat_train)

            logging.info(f"Calculating f1 test score")
            yhat_test = model.predict(x_test)
            f1_test_score = f1_score(y_true=y_test, y_pred=yhat_test)
            testing_accuracy = accuracy_score(y_true=y_test,y_pred=yhat_test)

            logging.info(f"train score:{training_accuracy} and test score: {testing_accuracy}")
            # logging.info(f"train f1:{f1_train_score} and test f1: {f1_test_score}")
            #check for overfitting and underfitting or expected score
            logging.info(f"Checking if the model is underfiting or not")
            if f1_test_score < self.model_trainer_config.excpected_score:
                raise Exception(f"Model is not good as it is unable to give \
                expected accuracy: {self.model_trainer_config.excpected_score}; actual model score:{f1_test_score}")

            logging.info(f"Checking if the model is overfitting or not")
            diff = abs(f1_train_score - f1_test_score)

            if diff > self.model_trainer_config.overfitting_threshold:
                raise Exception(f"Train and Test score diff: {diff} is more than Overfitting Threshold {self.model_trainer_config.overfitting_threshold}")
            
            #save the trained model
            logging.info(f"Saving Model object")
            utils.save_object(file_path=self.model_trainer_config.model_path, obj=model)

            #prepare artifact
            logging.info(f"Preparing the Artifact")
            model_trainer_artifact = artifact_entity.ModelTrainerArtifact(model_path=self.model_trainer_config.model_path,
            f1_train_score=f1_train_score, f1_test_score=f1_test_score)
            logging.info(f"Model Trainer Artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        except Exception as e:
            raise EllepticException(e, sys)
