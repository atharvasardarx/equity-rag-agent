import os
import glob
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

documents = []
# 1. Load PDFs
print("Loading PDFs...")
for pdf_file in glob.glob("./data/*.pdf"):
    loader = PyPDFLoader(pdf_file)
    documents.extend(loader.load())

# 2. Load CSVs
print("Loading CSVs...")
for csv_file in glob.glob("./data/*.csv"):
    loader = CSVLoader(csv_file)
    documents.extend(loader.load())

# 3. Split text into processing chunks
print("Splitting text...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)

# 4. Create embeddings and store in ChromaDB locally
print("Creating vector database...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_db = Chroma.from_documents(
    documents=chunks, 
    embedding=embeddings, 
    persist_directory="./chroma_db"
)

print("Success! Vector database created in ./chroma_db")