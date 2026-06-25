import pandas as pd
import os

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

# baca csv (gunakan path relatif)
csv_path = "data/sawit_qa.csv"
df = pd.read_csv(csv_path)

documents = []

# ubah csv jadi document
for _, row in df.iterrows():

    text = f"""
    Pertanyaan: {row['pertanyaan']}

    Jawaban: {row['jawaban']}
    """

    documents.append(
        Document(
            page_content=text,
            metadata={"id": row["id"]}
        )
    )

print(f"Jumlah data: {len(documents)}")

# chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

docs = text_splitter.split_documents(documents)

print(f"Jumlah chunk: {len(docs)}")

# embedding (gunakan Ollama seperti di app_csv.py)
embedding = OllamaEmbeddings(
    model="nomic-embed-text"
)

# simpan ke chromadb (ke db_csv seperti di app_csv.py)
db = Chroma.from_documents(
    docs,
    embedding,
    persist_directory="db_csv"
)

print("ChromaDB berhasil dibuat di db_csv!")