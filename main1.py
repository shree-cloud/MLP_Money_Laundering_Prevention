
from elleptic.pipeline.training_pipeline import initiate_training_pipeline
from elleptic.pipeline.batch_prediction import initiate_batch_prediction

file_path = "/config/workspace/MLP_training_dataset.csv"

if __name__=="__main__":
     try:
          # initiate_training_pipeline()
          output_file = initiate_batch_prediction(input_file_path=file_path)
          print(output_file)
     except Exception as e:
          print(e)
