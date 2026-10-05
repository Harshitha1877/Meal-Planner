import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma


# ==============================
# 1. PDF FILE LOCATION
# ==============================

PDF_PATH = "books/book1.pdf"

print("Starting RAG setup...")
print("Loading recipe book...")


# ==============================
# 2. CHECK PDF EXISTS
# ==============================

if not os.path.exists(PDF_PATH):
    print("ERROR: PDF file not found!")
    print("Make sure your file is here:")
    print(PDF_PATH)
    exit()


# ==============================
# 3. LOAD PDF
# ==============================

loader = PyPDFLoader(PDF_PATH)

documents = loader.load()

print("PDF loaded successfully!")
print("Number of pages:", len(documents))


# ==============================
# 4. SPLIT PDF INTO CHUNKS
# ==============================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)

print("Text chunks created:", len(chunks))


# ==============================
# 5. CREATE OLLAMA EMBEDDINGS
# ==============================

print("Creating embeddings...")

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

print("Embeddings model loaded successfully!")


# ==============================
# 6. CREATE VECTOR DATABASE
# ==============================

print("Creating recipe vector database...")

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="recipe_database"
)

print("Recipe database created successfully!")


# ==============================
# 7. FINISHED
# ==============================

print("--------------------------------")
print("RAG SETUP COMPLETED SUCCESSFULLY!")
print("--------------------------------")
print("Your recipe book is now ready.")
print("Database folder: recipe_database")