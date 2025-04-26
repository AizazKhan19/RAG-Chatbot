from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import gradio as gr
import os

# Load environment variables
load_dotenv()

# Paths
FAISS_INDEX_PATH = r"faiss_index"

# Load FAISS index
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS.load_local(FAISS_INDEX_PATH, embedding_model, allow_dangerous_deserialization=True)


# Convert FAISS to retriever
retriever = vector_store.as_retriever(search_kwargs={'k': 5})

# Use Free LLM from Hugging Face (Mistral-7B)
llm = HuggingFaceHub(
    repo_id="mistralai/Mistral-7B-Instruct-v0.1",
    model_kwargs={"temperature": 0.5, "max_length": 512},
)

# Create Retrieval-QA Chain
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Function to handle chatbot interaction
def chat(message, history):
    response = qa_chain.run(message)
    return response

# Gradio Chat Interface
chatbot = gr.ChatInterface(chat, textbox=gr.Textbox(placeholder="Enter your query..."))
chatbot.launch()
