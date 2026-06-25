import os
from google import genai

# kalau pakai .env
from dotenv import load_dotenv
load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

response = client.models.generate_content(
    model="gemini-1.5-flash",
    contents="Halo, apakah kamu aktif?"
)

print(response.text)