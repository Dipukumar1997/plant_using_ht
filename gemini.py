# import os
# import google.generativeai as genai
# from dotenv import load_dotenv

# load_dotenv()
# genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
# gemini_model = genai.GenerativeModel("gemini-1.5-flash")

# def get_gemini_response(prompt):
#     try:
#         response = gemini_model.generate_content(prompt)
#         return response.text.strip()
#     except Exception as e:
#         return f"❌ Gemini error: {e}"

# #emini_model = genai.GenerativeModel("gemini-1.5-flash")

import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Create model instance
gemini_model = genai.GenerativeModel("gemini-1.5-flash")

def get_gemini_response(prompt):
    """Returns Gemini response for any prompt."""
    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"❌ Gemini error: {e}"

def get_cure(disease_name):
    """Returns cure suggestion for a disease using Gemini."""
    prompt = f"Suggest a cure or treatment for the plant disease '{disease_name}' in simple terms."
    return get_gemini_response(prompt)
