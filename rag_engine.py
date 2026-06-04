import os
import sys

# FORCE ML CONFIGURATIONS BEFORE LOADING THIRDPARTY LIBRARIES
os.environ["AVOID_META_TENSORS"] = "True"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["USE_TF"] = "NO"
os.environ["USE_TORCH"] = "YES"

# Completely block TensorFlow from messing up protobuf bindings
sys.modules['tensorflow'] = None 

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

class UpworkRAGEngine:
    def __init__(self, file_path="API Documentation Partial.pdf", index_dir="faiss_upwork_index"):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.file_path = os.path.join(current_dir, file_path)
        self.index_dir = os.path.join(current_dir, index_dir)
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'device': 'cpu'}
        )
        self.vector_store = None

    def ingest_and_sanity_check(self):
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"Source baseline PDF missing at path: '{self.file_path}'")
        
        loader = PyPDFLoader(self.file_path)
        documents = loader.load()
        return documents

    def chunk_documents(self, documents):
        # OPTIMIZED: Larger windows keep related sentences and headers together
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_documents(documents)
        return chunks

    def build_vector_store(self, chunks):
        self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        self.vector_store.save_local(self.index_dir)
        return self.vector_store

    def load_local_store(self):
        if os.path.exists(self.index_dir):
            try:
                self.vector_store = FAISS.load_local(
                    self.index_dir, 
                    self.embeddings, 
                    allow_dangerous_deserialization=True
                )
                return True
            except Exception:
                return False
        return False

    def retrieve_context(self, query_text, k=5):
        if not self.vector_store:
            raise ValueError("Execution Aborted: Semantic lookup matrix uninitialized.")
        return self.vector_store.similarity_search(query_text, k=k)