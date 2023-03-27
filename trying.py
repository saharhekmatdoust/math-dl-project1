import streamlit as st
import pandas as pd
import os
import time
import numpy as np
import pandas as pd
from collections import namedtuple
from utils import openai_auth
from openai.embeddings_utils import get_embedding
from openai.embeddings_utils import cosine_similarity
import tiktoken

st.title("Looking for a job title description")

# adding a single-line text input widget
query = st.text_input('Job title')

st.write('You requested job description for job title: ', query)

DOCUMENTS_EMBEDDINGS_PATH = "data"  # a folder with all the documents embeddings. within this folder, one csv file include multiple documents embedding of the same run
COLUMN_EMBEDDINGS = "embedding"  # the embedding column name in the documents embedding file.

file_path = "data/TaskEmbeddingss.csv"
EMBEDDING_MODEL = "text-embedding-ada-002"


# read documents embeddings
df = pd.read_csv(file_path)
df[COLUMN_EMBEDDINGS] = df[COLUMN_EMBEDDINGS].apply(eval).apply(np.array)  # convert string to np array
st.write(f'Read {len(df)} documents embeddings from {file_path}')

openai_auth()

while True:
    query = st.text_input('Please enter a job title or type "exit" to exit')
    if query == "exit":
        break
    with st.spinner('Computing embeddings...'):
        tic = time.time()
        query_embedding = get_embedding(query, engine=EMBEDDING_MODEL)
        toc = time.time()
    st.write(f'Embedding query took {round(toc-tic)*1000}ms')
    st.write('Top matches ordered by cosine similarity of vector embeddings:')
    df['similarity'] = df[COLUMN_EMBEDDINGS].apply(lambda x: cosine_similarity(x, query_embedding))
    result = df.sort_values(by='similarity', ascending=False)[['similarity', 'description']].head(10)
    st.write(result)
