from elleptic.pipeline.training_pipeline import initiate_training_pipeline
from elleptic.pipeline.batch_prediction import initiate_batch_prediction

import streamlit as st
import pandas as pd
from datetime import datetime
import os
import time

if __name__=="__main__":
     try:
          st.title("Money Laundering Prediction")
          st.markdown("Upload the input data")
          
          if st.button("Initiate Training"):
               initiate_training_pipeline()


          uploaded_file = st.file_uploader("Upload Input File", type="csv")

          if uploaded_file is not None:
               st.write("Uploaded file:", uploaded_file.name)
               @st.cache_data
               def load_data():
                    df = pd.read_csv(uploaded_file)
                    return df
               
               df = load_data()

               if st.checkbox('show Raw Data', False):
                    st.subheader('Raw Data')
                    st.write(df)

               if st.button("Initiate Batch Prediction"):
                    output_file_path = initiate_batch_prediction(input_file_path=uploaded_file)
          
                    st.write("Batch Prediction Pipeline completed!")
          
                    with open(output_file_path, "rb") as file:
                         btn = st.download_button(
                              label="Download",
                              data=file,
                              file_name="modified_dataframe.csv",
                              mime="text/csv"
                         )
          
                    modified_dataframe = pd.read_csv(output_file_path)
                    if st.checkbox('Show Modified Data', False):
                         st.subheader('Modified Data')
                         st.write(modified_dataframe)

          
          print(output_file_path)
     except Exception as e:
          print(e)



               # output_file_path = initiate_batch_prediction(input_file_path=uploaded_file)

               # st.write("Batch Prediction Pipeline completed!")

               # with open(output_file_path, "rb") as file:
               #      btn = st.download_button(
               #           label="Download",
               #           data=file,
               #           file_name="modified_dataframe.csv",
               #           mime="text/csv"
               #      )
               

               # modified_dataframe = pd.read_csv(output_file_path)
               # if st.checkbox('show Modified Data', False):
               #      st.subheader('Modified Data')
               #      st.write(modified_dataframe)
