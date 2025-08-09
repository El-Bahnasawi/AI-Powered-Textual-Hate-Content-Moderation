from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = FastAPI()

tokenizer = AutoTokenizer.from_pretrained("medoxz543/hate-speech")
model = AutoModelForSequenceClassification.from_pretrained("medoxz543/hate-speech")
model.eval()

class TextRequest(BaseModel):
    texts: list[str]

@app.get("/")
def root():
    return {"status": "🟢 Hate Speech API is running!"}

@app.post("/check")
def check_text(payload: TextRequest):
    inputs = tokenizer(payload.texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[:, 1].tolist()
        return [{"score": round(p, 4), "blur": p > 0.4} for p in probs]
