import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("models/gemini-1.5-flash")

def get_fallback_response(question):
    try:
        response = model.generate_content(question)
        return response.text
    except Exception as e:
        print(f"Gemini Error: {e}")
        return "Sorry, I couldn't find anything related."

# Testing the function
if __name__ == "__main__":
    print("Testing Gemini response...")
    print(get_fallback_response("Who is Elon Musk?"))
