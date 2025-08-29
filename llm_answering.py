import os
from dotenv import load_dotenv
from ibm_watsonx_ai import APIClient

load_dotenv()

API_KEY = os.getenv("WATSONX_APIKEY")
PROJECT_ID = os.getenv("PROJECT_ID")
MODEL_ID = os.getenv("MODEL_ID", "ibm/granite-13b-instruct-v2")
ENDPOINT = os.getenv("ENDPOINT")

if not all([API_KEY, PROJECT_ID, ENDPOINT]):
    raise ValueError("❌ Missing one or more required environment variables.")

client = APIClient(
    params={
        "url": ENDPOINT,
        "apikey": API_KEY
    }
)
client.set.default_project(PROJECT_ID)

def generate_answer(context_chunks, question):
    context = "\n".join(context_chunks)
    prompt = f"""You are StudyMate, an academic assistant.

Context:
{context}

Question:
{question}

Answer:"""
    
    try:
        response = client.foundation_models.generate_text(
            model_id=MODEL_ID,
            prompt=prompt,
            temperature=0.5,
            max_new_tokens=500
        )
        print("DEBUG response:", response)  # temporary debug
        answer = response['results'][0]['generated_text']
    except Exception as e:
        answer = f"⚠️ Error generating answer: {str(e)}"
    
    return answer
