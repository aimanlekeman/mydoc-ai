
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types


# Load the API key from .env
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError("GEMINI_API_KEY is missing from .env")


# Set a 20-second timeout.
client = genai.Client(
    api_key=api_key,
    http_options=types.HttpOptions(timeout=60000)
)

print("Connecting to Gemini...")

try:
    response = client.models.generate_content(
        model="gemini-3.5-flash-lite",
        contents="Reply with exactly: Gemini connection successful.",
        config=types.GenerateContentConfig(
            max_output_tokens=30
        )
    )

    print("\nGemini response:")
    print(response.text)

except Exception as error:
    print("\nGemini connection failed!")
    print("Error type:", type(error).__name__)
    print("Error details:", str(error))
