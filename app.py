import streamlit as st 
import pandas as pd
import os
import matplotlib
from dotenv import load_dotenv
from pandasai.llm import AzureOpenAI
from pandasai import SmartDatalake
load_dotenv()
matplotlib.use('TkAgg')

st.title("Cengage DataAnalyzer Demo")

uploaded_files = st.file_uploader("Upload a CSV file for analysis", type=['csv','xlsx','sql'], accept_multiple_files=True)

dataframes = []
for file in uploaded_files:
    if file.name.endswith("csv"):
        dataframes.append(pd.read_csv(file))
    elif file.name.endswith("xlsx"):
        dataframes.append(pd.read_excel(file))
    elif file.name.endswith("sql"):
        dataframes.append(pd.read_sql(file))
    st.write(dataframes[-1].head(3))

prompt = st.text_area("Enter your prompt:")

llm = AzureOpenAI(
    deployment_name=os.environ['AZURE_OPENAI_MODEL'],
    api_base=os.environ['AZURE_OPENAI_ENDPOINT'],
    api_version="2023-05-15",
    is_chat_model=True,  # Comment in if you deployed a completion model
)

# Generate output
data = SmartDatalake(dataframes, config={"llm": llm})

if st.button("Generate"):
    if prompt:
        with st.spinner("Generating response..."):
                st.write(data.chat(prompt))
    else:
        st.warning("Please enter a prompt.")