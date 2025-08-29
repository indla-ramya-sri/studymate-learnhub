import requests
import os
from dotenv import load_dotenv

# Load .env values
load_dotenv()
API_KEY = os.getenv("WATSONX_APIKEY")

url = "https://iam.cloud.ibm.com/identity/token"
payload = {
    "apikey": API_KEY,
    "grant_type": "urn:ibm:params:oauth:grant-type:apikey"
}
headers = {"Content-Type": "application/x-www-form-urlencoded"}

resp = requests.post(url, data=payload, headers=headers)
print("Status:", resp.status_code)
print("Response:", resp.text[:500])  # print first 500 chars
