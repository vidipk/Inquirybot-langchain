import streamlit as st
import os
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
#from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# NEW IMPORT PATHS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# Load environment
load_dotenv()

# Page config
st.set_page_config(page_title="RAG Chatbot", layout="wide")
st.title("Ask Me Anything!")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

# Sidebar
with st.sidebar:
    st.header("Settings")
    
    # Check if Chroma DB exists
    chroma_path = "chroma_db"
    
    if st.button("📥 Ingest Documents"):
        with st.spinner("Ingesting documents..."):
            try:
                # Load documents
                loader = DirectoryLoader(
                    "data",
                    glob="**/*.pdf",
                    loader_cls=PyPDFLoader
                )
                documents = loader.load()
                
                if not documents:
                    st.error("No PDF files found in data/ folder")
                else:
                    # Split documents
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=200
                    )
                    chunks = text_splitter.split_documents(documents)
                    
                    # Create embeddings and store in Chroma
                    #
                    embeddings = OpenAIEmbeddings()
                    
                    vector_store = Chroma.from_documents(
                        documents=chunks,
                        embedding=embeddings,
                        persist_directory=chroma_path
                    )
                    
                    st.session_state.vector_store = vector_store
                    st.success(f"✅ Ingested {len(chunks)} document chunks!")
                    
            except Exception as e:
                st.error(f"Error ingesting documents: {e}")
    
    st.divider()
    
    if st.button("🗑️ Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# Load or create vector store
if st.session_state.vector_store is None:
    try:
        embeddings = OpenAIEmbeddings()
        st.session_state.vector_store = Chroma(
            persist_directory="chroma_db",
            embedding_function=embeddings
        )
    except:
        st.warning("⚠️ No vector store found. Please ingest documents first using the sidebar button.")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        try:
            if st.session_state.vector_store is None:
                st.error("No documents ingested. Please use the sidebar to ingest documents.")
            else:
                with st.spinner("Thinking..."):
                    # Setup RAG chain
                    embeddings = OpenAIEmbeddings()
                    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)
                    
                    # Retrieve relevant documents
                    retriever = st.session_state.vector_store.as_retriever(
                        search_kwargs={"k": 3}
                    )
                    
                    # Create prompt template
                    template = """You are a helpful assistant. Answer the question based only on the provided context.

Context:
{context}

Question: {question}

Answer:"""
                    
                    prompt_template = ChatPromptTemplate.from_template(template)
                    
                    # Create RAG chain
                    chain = (
                        {"context": retriever, "question": RunnablePassthrough()}
                        | prompt_template
                        | llm
                        | StrOutputParser()
                    )
                    
                    # Get response
                    response = chain.invoke(prompt)
                    
                    st.markdown(response)
                    
                    # Add assistant message to history
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response}
                    )
                    
        except Exception as e:
            st.error(f"Error generating response: {e}")
            st.error("Make sure your OPENAI_API_KEY is set correctly.")