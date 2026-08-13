import streamlit as st
from langchain_ollama import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langgraph.prebuilt import create_react_agent
from langchain.tools import tool
from sentiment import get_financial_sentiment

# Page Configuration
st.set_page_config(
    page_title="Equity RAG Agent",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Modern Dark/Light Aesthetic
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stTextInput input {
        background-color: #1e2530;
        color: #ffffff;
        border-radius: 8px;
        border: 1px solid #30363d;
    }
    .stChatMessage {
        border-radius: 12px;
        padding: 10px;
        margin-bottom: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar Design
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/stocks.png", width=100)
    st.title("Financial Intel")
    st.markdown("---")
    st.markdown("**Core Stack:**")
    st.code("LangChain • Ollama • FinBERT • ChromaDB")
    st.markdown("---")
    st.info("Tip: Ensure Ollama is running locally in the background before querying.")

# Main Header
st.title("📈 Agentic Equity & Sentiment Analyst")
st.markdown("Query dense corporate filings, annual shareholder letters, and trade logs with localized AI reasoning.")

# --- TOOLS ---
@tool
def retrieve_financial_docs(query: str) -> str:
    """Use this to search the local database for financial reports and context."""
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    docs = db.similarity_search(query, k=3)
    if not docs:
        return "No relevant documents found."
    return "\n".join([doc.page_content for doc in docs])

@tool
def analyze_sentiment(text: str) -> str:
    """Use this tool to analyze the financial sentiment of a given text."""
    return get_financial_sentiment(text)

SYSTEM_PROMPT = (
    "You are a helpful financial research assistant. Use the retrieve_financial_docs "
    "tool to search local filings/reports when the user asks about specific companies, "
    "numbers, or documents. Use analyze_sentiment when asked about tone or sentiment of "
    "a piece of text. Respond directly and concisely when no tool is needed."
)

@st.cache_resource
def load_agent():
    llm = ChatOllama(model="llama3.2:3b", temperature=0)
    tools = [retrieve_financial_docs, analyze_sentiment]
    agent_executor = create_react_agent(llm, tools, prompt=SYSTEM_PROMPT)
    return agent_executor

# --- CHAT UI ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("Ask about a filing, ticker, or sentiment..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            agent_executor = load_agent()
            response = agent_executor.invoke({"messages": [("user", user_input)]})
            answer = response["messages"][-1].content
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})