import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from transformers import AutoTokenizer

DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "utils")
)

class Encoder: 
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L12-v2", device="cpu" ): 
        
        self.embedding = HuggingFaceEmbeddings(model_name = model_name, 
                                                cache_dir=DIR, 
                                                model_kwargs = {"device": device})
        

class FaissDb: 
    def __init__(self, docs, embedding_function): 
        self.db = FAISS.from_documents(docs, embedding_function, distance_strategy=DistanceStrategy.COSINE)
        
    def similarity_search(self, query, top_k=5): 
        retrivied_docs = self.db.similarity_search(query, top_k)
        context = "".join(doc.page_content + "\n" for doc in retrivied_docs)
        
        return context
    
    
def load_and_split(file_path: list, chunk_size: int = 256): 
    loaders = [PyPDFLoader(file_path) for file_path in file_path]
    
    docs = []
    for loader in loaders: 
        docs.extend(loader.load())
        
    splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L12-v2"), 
        chunk_size = chunk_size,
        chunk_overlap = chunk_size // 10, 
        strip_whitespace = True,
    )
    
    docs_2 = splitter.split(docs)
    return docs_2


def save_file(uploaded_file, FILES_DIR): 
    file_path = os.path.join(FILES_DIR, uploaded_file.name)
    with open(file_path, "wb") as f: 
        f.write(uploaded_file.getbuffer())
    return file_path