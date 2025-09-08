# ----------------- Imports -----------------
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import torch
from transformers import AutoTokenizer, AutoModel
import contractions
import re

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

# ----------------- Load Models -----------------
pipeline = joblib.load("career_pipeline.pkl")       # Your trained classifier
label_encoder = joblib.load("label_encoder.pkl")    # LabelEncoder
career_desc = joblib.load("career_desc.pkl")        # Career description dict

# Transformer model for embeddings
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to("cpu")  # ensure CPU

def embed_text(text_list):
    inputs = tokenizer(text_list, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state[:,0,:]
    return embeddings.cpu().numpy()  # convert to NumPy safely

# ----------------- FastAPI Setup -----------------
app = FastAPI(title="AI Career Counselor")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")  # put your new2.html here

# Input model
class Interest(BaseModel):
    text: str

# Serve HTML page
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("new2.html", {"request": request})

# Predict career API
# @app.post("/predict")
# def predict_career(interest: Interest):
#     user_text = preprocess_text(interest.text)
#     embedding = embed_text([user_text])
#     pred_label = pipeline.predict(embedding)[0]
#     career_name = label_encoder.inverse_transform([pred_label])[0]
#     description = career_desc.get(career_name, "No description available.")
#     return {"career": career_name, "description": description}


@app.post("/predict")
def predict_career(interest: Interest):
    user_text = preprocess_text(interest.text)
    embedding = embed_text([user_text])

    # Get prediction probabilities
    probs = pipeline.predict_proba(embedding)[0]

    # Sort classes by probability
    top_indices = probs.argsort()[-3:][::-1]   # 👈 top 3
    top_careers = []
    for idx in top_indices:
        career_name = label_encoder.inverse_transform([idx])[0]
        description = career_desc.get(career_name, "No description available.")
        top_careers.append({
            "career": career_name,
            "description": description
        })

    # Use the first recommendation's description as "Next Steps"
    next_steps = top_careers[0]["description"] if top_careers else "No next steps available."

    return {
        "recommendations": top_careers,
        "next_steps": next_steps
    }
# # ----------------- Run Uvicorn -----------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
