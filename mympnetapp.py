import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
import pandas as pd 
import os 
import time  


# Define dataset of sentences for MPNet
file_path_mpnet = "https://drive.google.com/file/d/1yZ0ZrV0XVkLuHWLOVjZg3CtjEMIWDAF4/view?usp=share_link"
df_mpnet= pd.read_csv(file_path_mpnet)
mpnet_embedding=np.array(df_mpnet['embeddings'])

# Set up Streamlit app
st.title("Searching for Job Responsibilities based on Job Title")

input_text = st.text_input("Enter a Job Title:")


# Define function to calculate embeddings
def get_embeddings_MPNet(text, model):
    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    MPNet_embedding=np.array(model.encode(text, show_progress_bar=True))
    return MPNet_embedding

def cosine_similarity_mpnet(a,b):
    dot_product = np.dot(a, b)
    magnitude_a = np.sqrt(np.dot(a, a))
    magnitude_b = np.sqrt(np.dot(b, b))
    cos_sim = dot_product / (magnitude_a * magnitude_b)
    return cos_sim


# Calculate embeddings and get similar sentences
if st.button("Calculate"):

    model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    embeddings = mpnet_embedding
    input_embedding = get_embeddings_MPNet(input_text, model)
    cosine_sims = [cosine_similarity_mpnet(input_embedding,embedding) for embedding in embeddings]
    top_10_indices = np.argsort(-np.array(cosine_sims))[:10]
    selected_rows = df_mpnet.loc[top_10_indices.tolist(), 'Task']
    for i,desc in enumerate(selected_rows):
        st.write("Job Responsibility Number",i+1,'is:\n',desc)
    



