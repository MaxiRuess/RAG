import streamlit as st
import os
from dotenv import load_dotenv
from langchain.llms import OpenAI 
import streamlit as st
from langchain.llms import OpenAI
from langchain_openai import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy



openai_key = os.getenv("OPENAI_KEY")

FILES_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "files")
)

def save_file(uploaded_file, FILES_DIR): 
    file_path = os.path.join(FILES_DIR, uploaded_file.name)
    with open(file_path, "wb") as f: 
        f.write(uploaded_file.getbuffer())
    return file_path

def load_and_split(file_path: list, chunk_size: int = 256): 
    loaders = [PyPDFLoader(file_path) for file_path in file_path]
    
    docs = []
    for loader in loaders: 
        docs.extend(loader.load())
        
    splitter = CharacterTextSplitter(chunk_size = chunk_size, chunk_overlap = chunk_size // 10, strip_whitespace = True)
    
    docs_2 = splitter.split(docs)
    return docs_2


class Encoder:
    def __init__(self, openai_api_key: str): 
        
        self.embedding = OpenAIEmbeddings(openai_api_key=openai_api_key)



class FaissDb: 
    def __init__(self, docs, embedding_function): 
        self.db = FAISS.from_documents(docs, embedding_function, distance_strategy=DistanceStrategy.COSINE)
        
    def similarity_search(self, query, top_k=5): 
        retrivied_docs = self.db.similarity_search(query, top_k)
        context = "".join(doc.page_content + "\n" for doc in retrivied_docs)
        
        return context








class Model_GPT:
    def __init__(self, model_name: str = "gpt-3.5-turbo", device="cpu"):
        self.model = ChatOpenAI(model_name=model_name, temperatur = 0)
        
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_key)
                
    def generate(self, files, openai_key, query_text):
        docs = load_and_split(files)
        
        db = Chroma.from_documents(docs, embedding_function = self.embeddings)
        
        retriver = db.as_retriever()
        
        qa = RetrievalQA.from_chain_type(llm = self.model, chain_type = "stuff", retriever = retriver)
        
        return qa.run
        
    

def load_model():
    model_name = st.sidebar.selectbox("Model", ["gpt-3.5-turbo", "gpt-3.5-turbo-finetuned"])
    model = Model_GPT(model_name = model_name)
    return model
            
        
    
    
    
