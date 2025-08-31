# ----------------- Imports -----------------
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import joblib

# ----------------- Load Joblib Models -----------------
pipeline = joblib.load("career_pipeline.pkl")
career_desc = joblib.load("career_desc.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# ----------------- FastAPI App -----------------
app = FastAPI(title="AI Career Counselor")

# Templates folder
templates = Jinja2Templates(directory="templates")  # Create 'templates' folder and put your index.html there

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
    user_text = interest.text
    pred_label = pipeline.predict([user_text])[0]
    career_name = label_encoder.inverse_transform([pred_label])[0]
    description = career_desc.get(career_name, "No description available.")
    return {"career": career_name, "description": description}

# ----------------- Run Uvicorn -----------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
