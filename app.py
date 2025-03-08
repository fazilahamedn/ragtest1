import streamlit as st
import os
import re
import time
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings  # Fixed import path
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
groq_api_key = os.environ.get('GROQ_API_KEY')  # Using get() to avoid KeyError
if not groq_api_key:
    st.error("Groq API key not found. Please check your .env file.")
    st.stop()


# Function to create and save embeddings
def create_and_save_embeddings():
    try:
        # Initialize embeddings model
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}  # Force CPU usage to avoid CUDA issues
        )
        # Create an empty FAISS index
        vectors = FAISS.from_embeddings([], [], embeddings)  # Create with empty lists
        vectors.save_local(faiss_index_path)
        return vectors
    except Exception as e:
        st.error(f"Error creating embeddings: {str(e)}")
        return None


# Function to load embeddings
def load_embeddings():
    if os.path.exists(faiss_index_path):
        try:
            st.info("Loading existing embeddings...")
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}  # Force CPU usage to avoid CUDA issues
            )
            vectors = FAISS.load_local(faiss_index_path, embeddings, allow_dangerous_deserialization=True)
            st.success("Embeddings loaded successfully!")
            return vectors
        except Exception as e:
            st.error(f"Error loading embeddings: {str(e)}")
            st.info("Creating new embeddings...")
            return create_and_save_embeddings()
    else:
        st.info("No existing embeddings found. Creating new embeddings...")
        return create_and_save_embeddings()


# Page title
st.markdown("<h1 style='text-align: center;'>RAG Demo</h1>", unsafe_allow_html=True)

# Initialize or load vector store
if "vectors" not in st.session_state:
    st.session_state.vectors = load_embeddings()

# Initialize submitted flag if it doesn't exist
if "submitted" not in st.session_state:
    st.session_state.submitted = False


# Function to handle form submission
def handle_submit():
    st.session_state.submitted = True


# Only proceed if embeddings are loaded
if st.session_state.vectors:
    try:
        # Initialize the LLM
        llm = ChatGroq(model="deepseek-r1-distill-qwen-32b")
        
        # Create prompt template - Fixed the context tags
        prompt = ChatPromptTemplate.from_template(
            """
            Answer the questions based on the provided context only.
            Please provide the most accurate response based on the question.
            <context>
            {context}
            </context>
            Question: {input}
            """
        )

        # Create the document chain
        document_chain = create_stuff_documents_chain(llm, prompt)

        # Create the retriever
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        # Create a form for user input
        with st.form(key="query_form"):
            user_prompt = st.text_input("Input your prompt here", key="user_input", max_chars=200,
                                    label_visibility="collapsed")
            submit_button = st.form_submit_button("Submit", on_click=handle_submit)

        # Handle the submission
        if st.session_state.submitted:
            if user_prompt:  # Only process if there's actual input
                with st.spinner("Generating response..."):
                    start = time.time()
                    try:
                        response = retrieval_chain.invoke({"input": user_prompt})
                        end = time.time()

                        # Remove the <think>...</think> section from the answer
                        cleaned_answer = re.sub(r'<think>.*?</think>', '', response['answer'], flags=re.DOTALL).strip()
                        
                        # Display the answer
                        st.write(cleaned_answer)
                        st.caption(f"Response time: {end - start:.2f} seconds")

                        # Show retrieved documents
                        with st.expander("Document Similarity Search"):
                            for doc in response.get("context", []):
                                st.write(doc.page_content)
                                st.write("--------------------------------")
                    except Exception as e:
                        st.error(f"Error generating response: {str(e)}")

            # Reset the submitted flag for the next interaction
            st.session_state.submitted = False
    except Exception as e:
        st.error(f"Error initializing LLM or chain: {str(e)}")
else:
    st.error("Failed to initialize vector store. Please check the logs for details.")
