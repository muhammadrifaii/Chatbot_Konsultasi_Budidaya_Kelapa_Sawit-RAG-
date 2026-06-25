from langchain_chroma import Chroma

from langchain_ollama import (
    OllamaEmbeddings,
    ChatOllama
)
import os
from pathlib import Path

# =========================
# EMBEDDING
# =========================
embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

# =========================
# LOAD DATABASE
# =========================
db = Chroma(
    persist_directory="db_csv",
    embedding_function=embeddings
)

retriever = db.as_retriever(
    search_kwargs={"k": 1}
)

# =========================
# LOAD MODEL
# =========================
llm = ChatOllama(
    model="bakllava",
    temperature=0.1
)

# =========================
# CHAT LOOP
# =========================
while True:

    user_input = input("\nPertanyaan atau Path Gambar (ketik 'exit' untuk keluar): ").strip()

    if user_input.lower() == "exit":
        break

    # Check if input is image path
    is_image = False
    if os.path.exists(user_input):
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
        if Path(user_input).suffix.lower() in image_extensions:
            is_image = True
            question = f"[Image dari {user_input}] Apa pertanyaan tentang gambar ini?"
        else:
            print("File bukan gambar. Coba lagi dengan file gambar atau pertanyaan text.")
            continue
    else:
        # Input adalah pertanyaan text
        question = user_input

    # =========================
    # RETRIEVAL
    # =========================
    docs = retriever.invoke(question)

    context = "\n\n".join([
        doc.page_content for doc in docs
    ])

    # =========================
    # PROMPT
    # =========================
    if is_image:
        prompt = f"""Anda adalah asisten konsultasi budidaya kelapa sawit dengan kemampuan vision. PENTING: Jawab HANYA dalam Bahasa Indonesia.

Konteks Pengetahuan:
{context}

Gambar dari: {user_input}

Lihat gambar tersebut dan jawab pertanyaan tentang gambar berdasarkan konteks di atas. Jika informasi tidak ada di konteks, katakan "Informasi tidak ditemukan pada knowledge base."

Jawaban:"""
    else:
        prompt = f"""Jawab pertanyaan berikut berdasarkan informasi di bawah HANYA dalam Bahasa Indonesia.

Informasi:
{context}

Pertanyaan: {question}

Jawaban (tanpa penjelasan):"""

    # =========================
    # GENERATE
    # =========================
    try:
        response = llm.invoke(prompt)
        print("\n=== JAWABAN ===")
        if hasattr(response, 'content'):
            answer = response.content.strip()
            print(answer)
        else:
            print(str(response))
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()