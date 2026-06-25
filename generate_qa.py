import gradio as gr
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama
import os
from pathlib import Path
import base64
import requests
import json

# =====================================
# LOAD EMBEDDING
# =====================================
print("Loading Embedding Model...")
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# =====================================
# LOAD CHROMADB
# =====================================
print("Loading ChromaDB...")
db = Chroma(
    persist_directory="db_csv",
    embedding_function=embeddings
)

retriever = db.as_retriever(search_kwargs={"k": 1})
print("ChromaDB Loaded Successfully")

# =====================================
# LOAD MODEL
# =====================================
print("Loading LLM Model...")
llm = ChatOllama(model="bakllava", temperature=0.1)

# =====================================
# CHAT FUNCTION
# =====================================
def chatbot(message, image):
    try:
        if not message or not message.strip():
            return "❌ Silakan masukkan pertanyaan."

        # =========================
        # RETRIEVAL
        # =========================
        docs = retriever.invoke(message)
        context = "\n\n".join([doc.page_content for doc in docs])

        # =========================
        # HANDLE IMAGE
        # =========================
        if image and os.path.exists(image):
            # Validasi file image
            image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp'}
            if Path(image).suffix.lower() not in image_extensions:
                return "❌ File harus berupa gambar (jpg, png, gif, bmp, webp)"
            
            # Convert image ke base64
            try:
                with open(image, 'rb') as img_file:
                    image_data = base64.b64encode(img_file.read()).decode('utf-8')
            except Exception as e:
                return f"❌ Error membaca gambar: {str(e)}"
            
            prompt = f"""Anda adalah ahli budidaya kelapa sawit dengan kemampuan vision. Jawab dalam Bahasa Indonesia.

Informasi dari knowledge base:
{context}

Pertanyaan: {message}

Analisis gambar dan jawab berdasarkan informasi di atas (tanpa penjelasan panjang):"""
            
            # Gunakan Ollama API langsung untuk image
            try:
                response = requests.post(
                    'http://localhost:11434/api/generate',
                    json={
                        'model': 'bakllava',
                        'prompt': prompt,
                        'images': [image_data],
                        'stream': False
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    answer = result.get('response', '').strip()
                else:
                    answer = f"❌ Error dari Ollama: {response.status_code}"
            except Exception as e:
                answer = f"❌ Error API Ollama: {str(e)}"
        else:
            prompt = f"""Jawab pertanyaan berikut berdasarkan informasi di bawah HANYA dalam Bahasa Indonesia.

Informasi:
{context}

Pertanyaan: {message}

Jawaban (tanpa penjelasan):"""

            # =========================
            # GENERATE (tanpa gambar)
            # =========================
            response = llm.invoke(prompt)
            
            if hasattr(response, 'content'):
                answer = response.content.strip()
            else:
                answer = str(response).strip()
        
        return answer if answer else "⚠️ Tidak ada jawaban yang dihasilkan. Coba pertanyaan lain."

    except Exception as e:
        return f"❌ Error: {str(e)}"


# =====================================
# UI
# =====================================
with gr.Blocks(theme=gr.themes.Soft(), title="Konsultasi Kelapa Sawit") as demo:

    gr.Markdown("""
# 🌴 Chatbot Konsultasi Budidaya Kelapa Sawit

### Fitur
✅ Tanya jawab budidaya sawit  
✅ Upload gambar dan analisis (opsional)  
✅ Knowledge Base 20 pertanyaan  
✅ Powered by LangChain + ChromaDB + Ollama

### Cara Pakai
1. Ketik pertanyaan tentang budidaya sawit
2. Opsional: Upload gambar untuk analisis lebih detail
3. Klik "Kirim" dan tunggu jawaban
""")

    with gr.Row():
        with gr.Column():
            question = gr.Textbox(
                label="🔹 Pertanyaan",
                placeholder="Contoh: Berapa dosis pupuk RP saat penanaman bibit sawit?",
                lines=2
            )

            image = gr.Image(
                type="filepath",
                label="📸 Upload Gambar (Opsional)",
                value=None
            )

            submit_btn = gr.Button("📤 Kirim", variant="primary")

        with gr.Column():
            answer = gr.Textbox(
                label="🔸 Jawaban AI",
                lines=10,
                interactive=False
            )

    # Event
    submit_btn.click(
        fn=chatbot,
        inputs=[question, image],
        outputs=answer
    )
    
    # Enter key support
    question.submit(
        fn=chatbot,
        inputs=[question, image],
        outputs=answer
    )

    # Example questions
    gr.Examples(
        examples=[
            ["Berapa dosis pupuk RP saat penanaman bibit sawit?", None],
            ["Pada bulan apa penanaman sawit di areal replanting dilakukan?", None],
            ["Kapan lubang tanam dibuat sebelum penanaman?", None],
            ["Hama apa saja yang umum menyerang bibit kelapa sawit?", None],
        ],
        inputs=[question, image]
    )

# =====================================
# RUN
# =====================================
if __name__ == "__main__":
    print("\n✅ Chatbot ready! Open browser at http://localhost:7860")
    demo.launch(share=False)