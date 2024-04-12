
from dotenv import load_dotenv
import os
from PyPDF2 import PdfReader
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub


#set_seed(42)

def get_pdf_text(docs):
    text = ""
    for doc in docs: 
        pdf_reader = PdfReader(doc)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector(chunks):
    embeddings = OpenAIEmbeddings()
    #embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    knowledgeBase = FAISS.from_texts(chunks, embeddings)
    return knowledgeBase


def get_conversion(knowledgeBase):
    llm = ChatOpenAI(
    )
    #llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conv_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever = knowledgeBase.as_retriever(),
        memory=memory
    )
    return conv_chain
    
def handle_user_question(user_question):

        
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
                

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main(): 
    load_dotenv()
    
    st.set_page_config(page_title="Chat with your PDFs", page_icon=":books:", layout="wide")
    st.write(css, unsafe_allow_html=True)
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
        
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
        
    st.header("Chat with your PDFs :books:")

    
    user_question = st.text_input("Ask a question")
    if user_question:
        handle_user_question(user_question)
        

    with st.sidebar:
        st.subheader("Upload your PDF :page_facing_up:")
        docs = st.file_uploader("Upload your PDF and Submit", type="pdf", accept_multiple_files=True)
        
        
        if st.button("Submit"): 
            with st.spinner("Processing PDFs"): 
                raw_text =  get_pdf_text(docs)
                
                chunks = get_text_chunks(raw_text)
                vector = get_vector(chunks)
                
                st.success("PDFs processed")
                
                st.session_state.conversation = get_conversion(vector)



if __name__ == "__main__":
    main()

        