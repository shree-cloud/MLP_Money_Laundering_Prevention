from elleptic.pipeline.training_pipeline import initiate_training_pipeline
from elleptic.pipeline.batch_prediction import initiate_batch_prediction

import streamlit as st
import pandas as pd
from datetime import datetime
import os
import time

if __name__=="__main__":
     try:
          col1, col2 = st.columns((2,1))
          col1.title("Money Laundering Prediction")
          
          
          if col2.button("Initiate Training"):
               initiate_training_pipeline()

               st.markdown("For Batch prediction **Reload**")
               if st.button("Reload"):
                    st.caching.clear_cache()
                    st.experimental_rerun()


          col1.markdown("Upload the input data")
          uploaded_file = col1.file_uploader("Upload Input File", type="csv")

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
                              file_name=f"{output_file_path}.csv",
                              mime="text/csv"
                         )
          # modified_dataframe
               modified_dataframe = pd.read_csv(output_file_path)
               # if st.checkbox('Show Modified Data', False):
               #      st.subheader('Modified Data')
               #      st.write(modified_dataframe)

          
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
