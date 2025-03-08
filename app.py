import streamlit as st
import os
import re
import time
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

embeddings_dir = "embeddings"
faiss_index_path = os.path.join(embeddings_dir, "faiss_index")

# Ensure embeddings directory exists
os.makedirs(embeddings_dir, exist_ok=True)

## Load the Groq API key
groq_api_key = os.getenv('GROQ_API_KEY')
if not groq_api_key:
    st.error("Groq API key not found. Please check your .env file.")

# Function to create and save embeddings
def create_and_save_embeddings():
    # Initialize embeddings model
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Create some initial dummy text for FAISS to store
    dummy_documents = [
        Document(page_content="This is a test document for initializing FAISS."),
        Document(page_content="Retrieval-Augmented Generation (RAG) improves response accuracy."),
        Document(page_content="FAISS is used for efficient similarity search.")
    ]

    # Split the dummy text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    final_documents = text_splitter.split_documents(dummy_documents)

    # Create vector store
    vectors = FAISS.from_documents(final_documents, embeddings)
    
    # Save the FAISS index
    vectors.save_local(faiss_index_path)

    return vectors

# Function to load embeddings
def load_embeddings():
    if os.path.exists(faiss_index_path):
        try:
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vectors = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)
            return vectors
        except Exception as e:
            st.error(f"Error loading embeddings: {e}")
            return create_and_save_embeddings()
    else:
        return create_and_save_embeddings()

# Initialize or load vector store
if "vectors" not in st.session_state:
    st.session_state.vectors = load_embeddings()

st.markdown("<h1 style='text-align: center;'>RAG Demo</h1>", unsafe_allow_html=True)

# Initialize the LLM
llm = ChatGroq(model="deepseek-r1-distill-qwen-32b")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    <context>
    Question: {input}
    """
)

# Create the document chain
document_chain = create_stuff_documents_chain(llm, prompt)

# Create the retriever
if st.session_state.vectors:
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # User input
    user_prompt = st.text_input("Input your prompt here", key="user_input", max_chars=200, label_visibility="collapsed")

    if user_prompt:
        with st.spinner("Generating response..."):
            start = time.time()
            response = retrieval_chain.invoke({"input": user_prompt})
            end = time.time()

            # Remove the <think>...</think> section from the answer
            cleaned_answer = re.sub(r'<think>.*?</think>', '', response['answer'], flags=re.DOTALL).strip()

            st.write(cleaned_answer)
            st.caption(f"Response time: {end - start:.2f} seconds")

            # Clear input after submission
            st.session_state.user_input = ""

            # With a streamlit expander
            with st.expander("Document Similarity Search"):
                for doc in response.get("context", []):
                    st.write(doc.page_content)
                    st.write("--------------------------------")
else:
    st.error("Failed to initialize vector store. Please check the logs for details.")
