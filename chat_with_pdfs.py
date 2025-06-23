import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
import os

# --- Configuration ---
# For local development, set your GOOGLE_API_KEY as an environment variable.
# Example: os.environ["GOOGLE_API_KEY"] = "YOUR_API_KEY_HERE"
# In Canvas, this will be automatically handled.
api_key = "AIzaSybuPbJ1e9JRgq5eRJrOsM560of0QZKq8A6" # Leave this empty; Canvas will inject the key at runtime for generative-language.googleapis.com

# --- Utility Functions ---

def get_pdf_text(pdf_docs):
    """
    Extracts text from a list of PDF documents.
    Args:
        pdf_docs (list): A list of uploaded PDF files (from st.file_uploader).
    Returns:
        str: Concatenated text from all PDF documents.
    """
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text() or "" # Handle pages with no extractable text
    return text

def get_text_chunks(text):
    """
    Splits a large string of text into smaller, overlapping chunks.
    Args:
        text (str): The input text to be chunked.
    Returns:
        list: A list of text chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,        # Max characters in each chunk
        chunk_overlap=200,      # Overlap between chunks for context
        length_function=len     # Use character length
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    """
    Converts text chunks into embeddings and stores them in a FAISS vector store.
    Args:
        text_chunks (list): A list of text chunks.
    Returns:
        FAISS: A FAISS vector store containing the embeddings.
    """
    st.info("Creating text embeddings (this might take a moment)...")
    # Initialize GoogleGenerativeAIEmbeddings for creating vector representations
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    
    # Create FAISS vector store from the chunks and their embeddings
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    st.success("Vector store created successfully!")
    return vector_store

def get_conversational_chain(vector_store):
    """
    Sets up a conversational retrieval chain using a Gemini LLM and memory.
    Args:
        vector_store (FAISS): The FAISS vector store for retrieval.
    Returns:
        ConversationalRetrievalChain: The LangChain conversational chain.
    """
    st.info("Setting up conversational chain...")
    # Initialize the ChatGoogleGenerativeAI model (using gemini-2.0-flash for broader compatibility)
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.7, google_api_key=api_key)
    
    # Use ConversationBufferMemory to store chat history, explicitly setting output_key
    # The LangChainDeprecationWarning about memory migration is a general heads-up for future API changes,
    # but these parameters are still valid for this specific memory class.
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True, output_key='answer')
    
    # Create a ConversationalRetrievalChain.
    # It takes user input, retrieves relevant documents, and generates a response.
    # 'verbose=True' can be useful for debugging in development.
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory,
        return_source_documents=True # Optionally return the source chunks
    )
    st.success("Chatbot ready! Ask your questions.")
    return conversation_chain

# --- New Function to handle direct internal queries (like explaining PDF process) ---
def get_internal_response(user_question):
    """
    Provides predefined responses for questions about the application's functionality.
    This avoids sending self-referential questions to the LLM.
    """
    question_lower = user_question.lower()
    
    if "how do you work with pdfs" in question_lower or \
       "explain about the pdf" in question_lower or \
       "what do you do with pdfs" in question_lower or \
       "how do you process pdfs" in question_lower or \
       "tell me about pdf handling" in question_lower:
        return (
            "This application works with PDFs by first extracting all text from them. "
            "Then, it splits this text into smaller, overlapping 'chunks'. "
            "Each chunk is converted into a numerical 'embedding' (a vector representation of its meaning). "
            "These embeddings are stored in a 'vector store' (FAISS). "
            "When you ask a question, the application finds the most relevant chunks from your PDFs using these embeddings "
            "and sends them as context to a powerful AI model (Gemini) to generate an accurate answer based on your documents."
        )
    return None # Return None if no internal response is found

def handle_user_input(user_question):
    """
    Handles user questions, prioritizing internal explanations, then conversational chain.
    Args:
        user_question (str): The question posed by the user.
    """
    # First, check for internal, self-referential questions
    internal_response = get_internal_response(user_question)
    if internal_response:
        st.session_state.chat_history.append({"role": "user", "content": user_question})
        st.session_state.chat_history.append({"role": "assistant", "content": internal_response})
        # Display chat messages immediately after internal response
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        return

    # If no internal response, proceed with the conversational chain
    if "conversation" not in st.session_state or st.session_state.conversation is None:
        st.error("Please process your PDFs first to enable the chat.")
        return

    # Call the conversational chain with the user's question
    with st.spinner("Thinking..."):
        try:
            # Replaced deprecated __call__ with .invoke()
            response = st.session_state.conversation.invoke({'question': user_question})
            st.session_state.chat_history.append({"role": "user", "content": user_question})
            st.session_state.chat_history.append({"role": "assistant", "content": response['answer']})
        except Exception as e:
            st.error(f"An error occurred while processing your question: {e}")
            st.session_state.chat_history.append({"role": "user", "content": user_question})
            st.session_state.chat_history.append({"role": "assistant", "content": "I apologize, but I couldn't process your request. Please try again."})

    # Display chat messages
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

# --- Streamlit UI ---

def main():
    st.set_page_config(page_title="Chat with Your PDFs", page_icon="📚")
    st.title("📚 Chat with Your PDFs")
    st.markdown("Upload your PDF documents, process them, and then ask questions!")

    # Initialize session state variables if they don't exist
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "processed_pdfs" not in st.session_state:
        st.session_state.processed_pdfs = False # Flag to indicate if PDFs are processed

    with st.sidebar:
        st.header("Your Documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click 'Process'", accept_multiple_files=True, type="pdf"
        )
        if st.button("Process Documents"):
            if pdf_docs:
                with st.spinner("Processing documents..."):
                    # 1. Get PDF text
                    raw_text = get_pdf_text(pdf_docs)
                    
                    # 2. Get text chunks
                    text_chunks = get_text_chunks(raw_text)
                    
                    # 3. Create vector store
                    vector_store = get_vector_store(text_chunks)
                    
                    # 4. Create conversational chain
                    st.session_state.conversation = get_conversational_chain(vector_store)
                    st.session_state.processed_pdfs = True
                    st.session_state.chat_history = [] # Clear history on new processing
            else:
                st.warning("Please upload at least one PDF document.")

    # --- Chat Interface ---
    if st.session_state.processed_pdfs:
        user_question = st.chat_input("Ask a question about your documents:")
        if user_question:
            handle_user_input(user_question)
    else:
        st.info("Upload and process your PDF documents in the sidebar to start chatting!")

if __name__ == "__main__":
    main()
