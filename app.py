# ----------------- Imports -----------------
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import contractions
import re
import requests
import numpy as np
from pyngrok import ngrok
import nest_asyncio

# ----------------- Fix asyncio for Colab -----------------
nest_asyncio.apply()

# ----------------- Preprocessing -----------------
abbrev_dict = {
    "ml": "machine learning",
    "ai": "artificial intelligence",
    "cs": "computer science",
    "ds": "data science",
    "dev": "developer",
    "prog": "programming",
    "bio": "biology",
    "maths": "mathematics"
}

def preprocess_text(text):
    if text is None:
        return ""
    text = contractions.fix(text)
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    words = [abbrev_dict.get(w, w) for w in text.split()]
    return " ".join(words)

# ----------------- Load lightweight models -----------------
pipeline = joblib.load("career_pipeline.pkl")
label_encoder = joblib.load("label_encoder.pkl")
career_desc = joblib.load("career_desc.pkl")

# ----------------- API-based embeddings -----------------
EMBED_API = "https://api.sentence-transformers.org/embeddings"
API_KEY = "YOUR_API_KEY_HERE"  # Get free API key from SentenceTransformers or OpenAI

def embed_text(text_list):
    """
    Sends text to embeddings API and returns numpy embeddings.
    """
    response = requests.post(
        EMBED_API,
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={"texts": text_list}
    )
    response.raise_for_status()
    embeddings = np.array(response.json()["embeddings"])
    return embeddings

# ----------------- FastAPI Setup -----------------
app = FastAPI(title="AI Career Counselor")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")  # Place your new2.html here

# Input model
class Interest(BaseModel):
    text: str

# Serve HTML page
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("new2.html", {"request": request})

# Predict career API
@app.post("/predict")
def predict_career(interest: Interest):
    user_text = preprocess_text(interest.text)
    embedding = embed_text([user_text])
    pred_label = pipeline.predict(embedding)[0]
    career_name = label_encoder.inverse_transform([pred_label])[0]
    description = career_desc.get(career_name, "No description available.")
    return {"career": career_name, "description": description}

# ----------------- Start ngrok tunnel -----------------
public_url = ngrok.connect(8000)
print("Public URL:", public_url)

# ----------------- Run FastAPI -----------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
