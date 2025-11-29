"""Speech-to-text utilities for audio transcription and patient data extraction."""

import base64
import json
import os
import tempfile

import requests
from pydub import AudioSegment


def stt(audio_path: str, api_key: str, site_url: str = "https://medicine-assistant.app", 
        site_name: str = "Medicine Assistant") -> dict:
    """
    Speech-to-text function that processes audio and extracts patient data.
    
    Args:
        audio_path: Path to the audio file to process
        api_key: OpenRouter API key for Voxtral model
        site_url: Site URL for API headers
        site_name: Site name for API headers
    
    Returns:
        Dictionary containing extracted patient data fields for consult form:
        - HbA1c: Latest HbA1c value
        - Blood Glucose: Blood glucose level in mg/dL
        - eGFR: eGFR value in ml/min
        - Lipid Panel: Lipid panel results
        - Blood Pressure: Blood pressure reading
        - Symptoms & Notes: Any symptoms or notes mentioned
        - Treatments Adjustments: Treatment adjustment information
    """
    # Validate audio file exists
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file '{audio_path}' not found.")
    
    # Compress audio and encode to base64
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
        compressed_path = tmp_file.name
    
    try:
        # Load and compress audio
        audio_segment = AudioSegment.from_file(audio_path)
        audio_segment.export(compressed_path, format="mp3", bitrate="64k")
        
        # Read compressed file and encode to base64
        with open(compressed_path, "rb") as f:
            audio_base64 = base64.b64encode(f.read()).decode('utf-8')
        
        # Call Voxtral API
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": site_url,
                "X-Title": site_name,
            },
            json={
                "model": "mistralai/voxtral-small-24b-2507",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "Analyze the audio and extract patient medical data. "
                                    "Return ONLY a JSON object with these exact keys (use empty string if not mentioned):\n"
                                    '{"HbA1c": "", "Blood Glucose": "", "eGFR": "", '
                                    '"Lipid Panel": "", "Blood Pressure": "", '
                                    '"Symptoms & Notes": "", "Treatments Adjustments": ""}'
                                )
                            },
                            {
                                "type": "input_audio",
                                "input_audio": {
                                    "data": audio_base64,
                                    "format": "mp3"
                                }
                            }
                        ]
                    }
                ]
            },
            timeout=120
        )
        
        response.raise_for_status()
        result = response.json()
        
        # Extract content from response
        content = result['choices'][0]['message']['content']
        
        # Parse JSON from response (handle markdown code blocks if present)
        content = content.strip()
        if content.startswith("```"):
            # Remove markdown code block
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()
        
        data = json.loads(content)
        
        # Ensure all expected keys exist with default empty string
        return {
            "HbA1c": data.get("HbA1c", ""),
            "Blood Glucose": data.get("Blood Glucose", ""),
            "eGFR": data.get("eGFR", ""),
            "Lipid Panel": data.get("Lipid Panel", ""),
            "Blood Pressure": data.get("Blood Pressure", ""),
            "Symptoms & Notes": data.get("Symptoms & Notes", ""),
            "Treatments Adjustments": data.get("Treatments Adjustments", "")
        }
        
    finally:
        # Clean up temporary file
        if os.path.exists(compressed_path):
            os.remove(compressed_path)