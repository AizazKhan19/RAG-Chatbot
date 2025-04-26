from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS  # FAISS Import
from langchain.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Configuration paths
DATA_PATH = r"data"        # Folder containing PDFs
FAISS_INDEX_PATH = r"faiss_index"  # Folder to save FAISS index

# Initialize HuggingFaceEmbeddings model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load the PDF documents
loader = PyPDFDirectoryLoader(DATA_PATH)
raw_documents = loader.load()

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=100,
    length_function=len,
    is_separator_regex=False,
)

chunks = text_splitter.split_documents(raw_documents)

# Convert document chunks into plain text
chunk_texts = [chunk.page_content for chunk in chunks]

# Create FAISS index
vector_store = FAISS.from_texts(texts=chunk_texts, embedding=embedding_model)

# Save FAISS index
vector_store.save_local(FAISS_INDEX_PATH)

print("FAISS Index created and saved successfully!")
