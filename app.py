# ----------------- Imports -----------------
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import re
import contractions

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

def preprocess_text(text: str) -> str:
    """Clean and normalize user text input"""
    if not text:
        return ""
    text = contractions.fix(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    words = [abbrev_dict.get(w, w) for w in text.split()]
    return " ".join(words)

# ----------------- Load Models -----------------
pipeline = joblib.load("career_pipeline.pkl")
label_encoder = joblib.load("label_encoder.pkl")
career_desc = joblib.load("career_desc.pkl")

# ----------------- FastAPI Setup -----------------
app = FastAPI(title="AI Career Counselor")

# Enable CORS (important for frontend integration)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Templates folder
templates = Jinja2Templates(directory="templates")

# Input model
class Interest(BaseModel):
    text: str

# ----------------- Routes -----------------
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("new2.html", {"request": request})

@app.post("/predict")
def predict_career(interest: Interest):
    user_text = preprocess_text(interest.text)
    pred_label = pipeline.predict([user_text])[0]
    career_name = label_encoder.inverse_transform([pred_label])[0]
    description = career_desc.get(career_name, "No description available.")
    return {"career": career_name, "description": description}

# ----------------- Run App -----------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
