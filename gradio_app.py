import gradio as gr
from google import genai
from dotenv import load_dotenv
import os
from pathlib import Path
from langchain_chroma import Chroma
from PIL import Image
from langchain_ollama import OllamaEmbeddings

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError(
        "❌ GEMINI_API_KEY tidak ditemukan. "
        "Pastikan ada file .env (titik di depan, bukan _env) "
        "di folder yang sama dengan script ini, berisi:\n"
        "GEMINI_API_KEY=your_api_key_here"
    )

client = genai.Client(api_key=GEMINI_API_KEY)

# =====================================
# LOAD EMBEDDING
# =====================================
print("Loading Embedding Model...")
embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

# =====================================
# LOAD CHROMADB
# =====================================
print("Loading ChromaDB...")
db = Chroma(
    persist_directory="db_csv",
    embedding_function=embeddings
)

retriever = db.as_retriever(
    search_kwargs={"k": 8}
)
print("ChromaDB Loaded Successfully")

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

        context = "\n\n".join([
            doc.page_content
            for doc in docs
        ])

        # Referensi
        references = []

        for i, doc in enumerate(docs):
            source = doc.metadata.get(
                "id",
                f"Dokumen {i+1}"
            )

            references.append(
                f"- {source}"
            )

        prompt = f"""
Anda adalah pakar budidaya kelapa sawit.

Gunakan informasi berikut sebagai sumber utama.

========================
KNOWLEDGE BASE
========================

{context}

========================
PERTANYAAN
========================

{message}

Instruksi:

1. Jawab dalam Bahasa Indonesia.
2. Berikan jawaban lengkap.
3. Jelaskan langkah demi langkah jika diperlukan.
4. Jangan mengarang.
5. Jika informasi tidak ditemukan, katakan dengan jujur.
6. Tambahkan kesimpulan di akhir.
"""

        # =========================
        # DENGAN GAMBAR
        # =========================

        if image and os.path.exists(image):

            img = Image.open(image)

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[
                    prompt,
                    img
                ]
            )

        # =========================
        # TANPA GAMBAR
        # =========================

        else:

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )

        answer = response.text

        if references:

            answer += "\n\n📚 Referensi Knowledge Base\n"

            answer += "\n".join(
                list(set(references))
            )

        return answer

    except Exception as e:

        return f"❌ Error: {str(e)}"

# =====================================
# UI
# =====================================
with gr.Blocks(theme=gr.themes.Soft(primary_hue="green"), title="Konsultasi Kelapa Sawit") as demo:

    gr.Markdown("""
#  Chatbot Konsultasi Budidaya Kelapa Sawit
Tanya jawab seputar budidaya sawit, bisa juga upload gambar untuk analisis.
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