import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

st.markdown("""
    <style>
    /* Global App Styling */
    .stApp {
        background-color: #0A0A0A; /* Dark background */
        color: #FFFFFF; /* White text */
        font-family: 'Roboto', sans-serif; /* Futuristic font */
    }
    
    /* Chat Input Styling */
    .stChatInput input {
        background-color: #1A1A1A !important; /* Dark input background */
        color: #FFFFFF !important; /* White text */
        border: 1px solid #00FFAA !important; /* Neon green border */
        border-radius: 12px !important; /* Rounded corners */
        padding: 10px 15px !important; /* Padding for better spacing */
        font-size: 16px !important; /* Larger font size */
        transition: all 0.3s ease !important; /* Smooth transition */
    }
    
    .stChatInput input:focus {
        border-color: #00FFAA !important; /* Neon green border on focus */
        box-shadow: 0 0 8px rgba(0, 255, 170, 0.6) !important; /* Glow effect */
    }
    
    /* User Message Styling */
    .stChatMessage[data-testid="stChatMessage"]:nth-child(odd) {
        background-color: #1A1A1A !important; /* Dark background */
        border: 1px solid #00FFAA !important; /* Neon green border */
        color: #E0E0E0 !important; /* Light gray text */
        border-radius: 15px !important; /* Rounded corners */
        padding: 15px !important; /* Padding */
        margin: 10px 0 !important; /* Margin */
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important; /* Subtle shadow */
        transition: all 0.3s ease !important; /* Smooth transition */
    }
    
    /* Assistant Message Styling */
    .stChatMessage[data-testid="stChatMessage"]:nth-child(even) {
        background-color: #2A2A2A !important; /* Slightly lighter background */
        border: 1px solid #00FFAA !important; /* Neon green border */
        color: #F0F0F0 !important; /* Light gray text */
        border-radius: 15px !important; /* Rounded corners */
        padding: 15px !important; /* Padding */
        margin: 10px 0 !important; /* Margin */
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important; /* Subtle shadow */
        transition: all 0.3s ease !important; /* Smooth transition */
    }
    
    /* Avatar Styling */
    .stChatMessage .avatar {
        background-color: #00FFAA !important; /* Neon green background */
        color: #000000 !important; /* Black text */
        border-radius: 50% !important; /* Circular avatar */
        padding: 10px !important; /* Padding */
        font-size: 18px !important; /* Larger font size */
        box-shadow: 0 0 10px rgba(0, 255, 170, 0.6) !important; /* Glow effect */
    }
    
    /* Text Color Fix */
    .stChatMessage p, .stChatMessage div {
        color: #FFFFFF !important; /* White text */
    }
    
    /* File Uploader Styling */
    .stFileUploader {
        background-color: #1A1A1A !important; /* Dark background */
        border: 1px solid #00FFAA !important; /* Neon green border */
        border-radius: 12px !important; /* Rounded corners */
        padding: 15px !important; /* Padding */
        margin: 10px 0 !important; /* Margin */
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important; /* Subtle shadow */
    }
    
    /* Headings Styling */
    h1, h2, h3 {
        color: #00FFAA !important; /* Neon green text */
        font-family: 'Orbitron', sans-serif !important; /* Futuristic font */
        text-shadow: 0 0 5px rgba(0, 255, 170, 0.6) !important; /* Glow effect */
    }
    
    /* Buttons Styling */
    .stButton button {
        background-color: #00FFAA !important; /* Neon green background */
        color: #000000 !important; /* Black text */
        border: none !important; /* No border */
        border-radius: 12px !important; /* Rounded corners */
        padding: 10px 20px !important; /* Padding */
        font-size: 16px !important; /* Larger font size */
        font-weight: bold !important; /* Bold text */
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important; /* Subtle shadow */
        transition: all 0.3s ease !important; /* Smooth transition */
    }
    
    .stButton button:hover {
        background-color: #00CC88 !important; /* Darker green on hover */
        box-shadow: 0 0 10px rgba(0, 255, 170, 0.6) !important; /* Glow effect */
    }
    
    /* Sidebar Styling */
    .stSidebar {
        background-color: #1A1A1A !important; /* Dark background */
        border-right: 1px solid #00FFAA !important; /* Neon green border */
    }
    
    .stSidebar .stButton button {
        background-color: #1A1A1A !important; /* Dark background */
        color: #00FFAA !important; /* Neon green text */
        border: 1px solid #00FFAA !important; /* Neon green border */
    }
    
    .stSidebar .stButton button:hover {
        background-color: #00FFAA !important; /* Neon green background on hover */
        color: #000000 !important; /* Black text */
    }
    
    /* Add a futuristic font */
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&family=Roboto:wght@400;500&display=swap');
    </style>
    """, unsafe_allow_html=True)

PROMPT_TEMPLATE = """
You are an expert research assistant with extensive experience in simplifying complex research concepts for a wide audience. Your goal is to help people understand various research topics in a very simple and accessible manner.

Your task is to answer the following research-related query.While responding, please ensure that you break down the concepts into easy-to-understand terms, avoiding jargon and technical language. Keep your explanations concise and relatable, using analogies or examples where necessary to enhance understanding. If you are unsure about any aspect of the query, clearly state that you don't know.

Query: {user_query} 
Context: {document_context} 
Answer:
"""
PDF_STORAGE_PATH = '/Users/shreehari/Documents/GenAi projects/Gen-AI-With-Deep-Seek-R1-main/pdfs'
EMBEDDING_MODEL = OllamaEmbeddings(model="deepseek-r1:1.5b")
DOCUMENT_VECTOR_DB = InMemoryVectorStore(EMBEDDING_MODEL)
LANGUAGE_MODEL = OllamaLLM(model="deepseek-r1:1.5b")


def save_uploaded_file(uploaded_file):
    file_path = PDF_STORAGE_PATH + uploaded_file.name
    with open(file_path, "wb") as file:
        file.write(uploaded_file.getbuffer())
    return file_path

def load_pdf_documents(file_path):
    document_loader = PDFPlumberLoader(file_path)
    return document_loader.load()

def chunk_documents(raw_documents):
    text_processor = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    return text_processor.split_documents(raw_documents)

def index_documents(document_chunks):
    DOCUMENT_VECTOR_DB.add_documents(document_chunks)

def find_related_documents(query):
    return DOCUMENT_VECTOR_DB.similarity_search(query)

def generate_answer(user_query, context_documents):
    context_text = "\n\n".join([doc.page_content for doc in context_documents])
    conversation_prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    response_chain = conversation_prompt | LANGUAGE_MODEL
    return response_chain.invoke({"user_query": user_query, "document_context": context_text})


# UI Configuration


st.title("Research Assistant")
st.markdown("### Understand your Research papers")
st.markdown("---")

# File Upload Section
uploaded_pdf = st.file_uploader(
    "Upload Research Document (PDF)",
    type="pdf",
    help="Select a PDF document for analysis",
    accept_multiple_files=False

)

if uploaded_pdf:
    saved_path = save_uploaded_file(uploaded_pdf)
    raw_docs = load_pdf_documents(saved_path)
    processed_chunks = chunk_documents(raw_docs)
    index_documents(processed_chunks)
    
    st.success("✅ Document processed successfully! Ask your questions below.")
    
    user_input = st.chat_input("Enter your question about the document...")
    
    if user_input:
        with st.chat_message("user"):
            st.write(user_input)
        
        with st.spinner("Analyzing document..."):
            relevant_docs = find_related_documents(user_input)
            ai_response = generate_answer(user_input, relevant_docs)
            
        with st.chat_message("assistant", avatar="🤖"):
            st.write(ai_response)
