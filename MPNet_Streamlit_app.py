import streamlit as st
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import ast
#import spacy

# Load a pre-trained model for word embeddings
model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
# Load your dataframe here
file_path_mpnet = "top500embeddings.csv"
df_mpnet= pd.read_csv(file_path_mpnet)
df_mpnet['embeddings '] = df_mpnet['embeddings '].apply(ast.literal_eval)


# Define a function to calculate the cosine similarity between two vectors
def cosine_similarity_mpnet(a,b):
    dot_product = np.dot(a, b)
    magnitude_a = np.sqrt(np.dot(a, a))
    magnitude_b = np.sqrt(np.dot(b, b))
    cos_sim = dot_product / (magnitude_a * magnitude_b)
    return cos_sim

# Define the Streamlit app
def app():
    # Add a title and description
    st.title("Job Responsibility Finder")
    st.write("Enter a Job Title and get The required Tasks for it.")

    # Add a text input for the user to enter their text
    input_text = st.text_input("Enter a Job Title")

    # Check if the user has entered any text
    if input_text:
        # Process the input text to get its embedding
        input_vector = model.encode(input_text) 

        # Calculate the cosine similarity between the input vector and all the vectors in the dataframe
        similarity_scores =  [cosine_similarity_mpnet(input_vector,embedding) for embedding in df_mpnet['embeddings '].tolist()]

        # Get the top 10 most similar rows 
        #top_10_indices = similarity_scores.argsort()[::-1][:10]
        top_10_indices = np.argsort(-np.array(similarity_scores))[:5]
        selected_rows =df_mpnet.loc[top_10_indices.tolist(), 'Task']
        #top_10_scores = similarity_scores.iloc[top_5_indices]
        #top_10_rows =  df_mpnet.iloc[top_5_indices]
        for i,desc in enumerate(selected_rows):
            st.write("Job Responsibility Number",i+1,'is:\n',desc)

        # Display the top 5 most similar rows
        #st.write("Top 5 most similar rows:")
        #for i, row in top_5_rows.iterrows():
         #   st.write(f"{i}. {row['Task']} (similarity score: {top_5_scores.loc[i]:.2f})")

# Run the app
if __name__ == "__main__":
    app()
