import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def translate_en_to_ar(text: str) -> str:
    model_name = "gemini-2.5-flash"
    model = genai.GenerativeModel(model_name)

    # Build the contents correctly for Gemini SDK
    prompt = (
        "You are a professional medical translator. Translate the following text into clear, "
        "fluent Modern Standard Arabic suitable for a patient to read and understand. "
        "Use simple and natural language while keeping medical information accurate. "
        "Translate medical terms into commonly used Arabic equivalents (for example, "
        "'HbA1c' → 'السكري التراكمي'). "
        "Make sentences short, clear, and easy for a non-medical reader to understand. "
        "Just write the translation without annotating that this text is being translated to Arabic or any other notations.\n\n"
        f"Text to translate:\n{text}"
    )

    contents = [
        {
            "parts": [
                {"text": prompt}
            ]
        }
    ]

    response = model.generate_content(contents=contents)
    return response.text.strip()