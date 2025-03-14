import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import os

load_dotenv()
GOOGLE_API_KEY = os.getenv("API_KEY")



def get_pdf_text(pdf_docs):
    """Extracts text from each uploaded PDF file separately."""
    pdf_texts = {}
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() or ""  # Handle cases where text extraction fails
        pdf_texts[pdf.name] = text  # Store text with filename as key
    return pdf_texts


def get_text_chunks(pdf_texts):
    """Splits extracted text into chunks for each document separately."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    all_chunks = []
    metadata_list = []

    for pdf_name, text in pdf_texts.items():
        chunks = text_splitter.split_text(text)
        all_chunks.extend(chunks)
        metadata_list.extend([{"source": pdf_name}] * len(chunks))  # Store filename as metadata

    return all_chunks, metadata_list


def get_vector_store(text_chunks, metadata_list):
    """Generates vector embeddings and stores them using FAISS."""
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    vector_store = FAISS.from_texts(text_chunks, embeddings, metadatas=metadata_list)
    return vector_store


def get_conversational_chain(vector_store):
    """Creates a conversational chain using Google Gemini AI."""
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", api_version="v1", google_api_key=GOOGLE_API_KEY)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vector_store.as_retriever(search_kwargs={"k": 5}), memory=memory
    )
    return conversation_chain


def user_input(user_question):
    """Handles user input and generates AI responses."""
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chatHistory = response['chat_history']

    for i, message in enumerate(st.session_state.chatHistory):
        if i % 2 == 0:
            st.write("Human: ", message.content)
        else:
            st.write("Bot: ", message.content)


def main():
    """Streamlit UI for AI-powered PDF Knowledge Assistant."""
    st.set_page_config(page_title="DocuQuery: AI-Powered PDF Knowledge Assistant")
    st.header("DocuQuery: AI-Powered PDF Knowledge Assistant")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chatHistory" not in st.session_state:
        st.session_state.chatHistory = None

    if user_question and st.session_state.conversation:
        user_input(user_question)

    with st.sidebar:
        st.title("Settings")
        st.subheader("Upload your Documents")

        pdf_docs = st.file_uploader(
            "Upload your PDF Files and Click on the Process Button",
            accept_multiple_files=True
        )

        if st.button("Process"):
            if not pdf_docs:
                st.error("Please upload at least one PDF file.")
            else:
                with st.spinner("Processing..."):
                    pdf_texts = get_pdf_text(pdf_docs)  # Extract text separately for each PDF
                    text_chunks, metadata_list = get_text_chunks(pdf_texts)  # Get chunks with metadata
                    vector_store = get_vector_store(text_chunks, metadata_list)  # Create vector store
                    st.session_state.conversation = get_conversational_chain(vector_store)  # Store conversation chain
                    st.success("Processing complete! You can now ask questions.")

if __name__ == "__main__":
    main()