import os
import streamlit as st
from model import Model
import utils as utils


FILES_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "files")
)

st.title("Chatbot")

def load_model(): 
    model_name = st.sidebar.selectbox("Model", ["google/gemma-2b-it", "facebook/blenderbot-400M-distill", "facebook/blenderbot-400M-distill-finetuned"])
    model = Model(model_id = model_name)
    return model


def load_encoder(): 
    model_name = st.sidebar.selectbox("Model", ["sentence-transformers/all-MiniLM-L12-v2", "sentence-transformers/all-MiniLM-L6-v2"])
    encoder = utils.Encoder(model_name = model_name)
    return encoder


model = load_model()
encoder = load_encoder()

with st.sidebar: 
    st.title("Upload Files")
    uploaded_files = st.file_uploader("Upload files", type=["pdf"], accept_multiple_files=True)

    max_tokens = st.slider("Max Tokens", min_value=50, max_value=500, value=250, step=50)
    k = st.slider("Top k", min_value=1, max_value=10, value=5, step=1)
        
    file_path = []
    
    for uploaded_file in uploaded_files: 
        file_path.append(save_file(uploaded_file, FILES_DIR))
    
    if uploaded_files != []: 
        docs = utils.load_and_split(file_path)
        DB = utils.FaissDb(docs = docs, embedding_function = encoder.embedding)
        
    
    