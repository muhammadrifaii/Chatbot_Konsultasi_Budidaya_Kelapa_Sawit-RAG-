from dotenv import load_dotenv
import os

print("CWD:", os.getcwd())
print(".env ditemukan?", os.path.exists(".env"))

result = load_dotenv()
print("load_dotenv berhasil?", result)

key = os.getenv("GEMINI_API_KEY")
print("KEY ditemukan:", key is not None)
print("Panjang key:", len(key) if key else 0)
