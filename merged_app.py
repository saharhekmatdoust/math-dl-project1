import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
import pandas as pd 
import plotly.express as px
import os 
import time 
#from collections import namedtuple
from utils import openai_auth
from openai.embeddings_utils import get_embedding
from openai.embeddings_utils import cosine_similarity
import ast
#import tiktoken
#import torch 


# Define list of available models
available_models = ["open AI API", "MPNet Sentence Transformer"]

# Define dataset of sentences for MPNet
file_path_mpnet = ".\Data\task_with_embedding.csv"
df_mpnet= pd.read_csv(file_path_mpnet)
df_mpnet['embeddings '] = df_mpnet['embeddings '].apply(ast.literal_eval)



# Define dataset of sentences for MPNet
DOCUMENTS_EMBEDDINGS_PATH = "data"  # a folder with all the documents embeddings. within this folder, one csv file include multiple documents embedding of the same run
COLUMN_EMBEDDINGS = "embedding"  # the embedding column name in the documents embedding file.
file_path = "data/TaskEmbeddingss.csv"
df_ai = pd.read_csv(file_path)
df_ai[COLUMN_EMBEDDINGS] = df_ai[COLUMN_EMBEDDINGS].apply(eval).apply(np.array)  # convert string to np array
#st.write(f'Read {len(df_ai)} documents embeddings from {file_path}')


# Set up Streamlit app
st.title("Searching for Job Responsibilities based on Job Title")

# Ask user to choose model
model_choice = st.selectbox("Choose a model:", available_models)
model_dict={"open AI API":"text-embedding-ada-002",
             "MPNet Sentence Transformer":SentenceTransformer('sentence-transformers/all-mpnet-base-v2')}
# Get user input
input_text = st.text_input("Enter a Job Title:")


def cosine_similarity_mpnet(a,b):
    dot_product = np.dot(a, b)
    magnitude_a = np.sqrt(np.dot(a, a))
    magnitude_b = np.sqrt(np.dot(b, b))
    cos_sim = dot_product / (magnitude_a * magnitude_b)
    return cos_sim


# Calculate embeddings and get similar sentences
if st.button("Calculate"):
    if model_choice=="MPNet Sentence Transformer":
        model = model_dict[model_choice]  # Replace with code to load selected model
        input_vector = model.encode(input_text)
        similarity_scores =  [cosine_similarity_mpnet(input_vector,embedding) for embedding in df_mpnet['embeddings '].tolist()]
        top_10_indices = np.argsort(-np.array(similarity_scores))[:10]
        selected_rows = df_mpnet.loc[top_10_indices.tolist(), 'Task']
        for i,desc in enumerate(selected_rows):
            st.write("Job Responsibility Number",i+1,'is:\n',desc)


    elif model_choice=="open AI API":
        openai_auth()
        with st.spinner('Computing embeddings...'):
            tic = time.time()
            query_embedding = get_embedding(input_text, engine=model_dict["open AI API"])
            toc = time.time()
        st.write(f'Embedding query took {round(toc-tic)*1000}ms')
        st.write('Top matches ordered by cosine similarity of vector embeddings:')
        df_ai['similarity'] = df_ai[COLUMN_EMBEDDINGS].apply(lambda x: cosine_similarity(x, query_embedding))
        result = df_ai.sort_values(by='similarity', ascending=False)[['similarity', 'description']].head(10)
        st.write(result)



