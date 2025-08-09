from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from datetime import datetime
import torch
import model_loader
import os
from rule_based_filter import debug_rule_based_check


# Optional Supabase logging
SUPABASE_ENABLED = os.getenv("SUPABASE_URL") and os.getenv("SUPABASE_KEY")
if SUPABASE_ENABLED:
    from logger import sync_log_to_db

router = APIRouter()
DEVICE = model_loader.DEVICE

# Input schema
class TextRequest(BaseModel):
    texts: list[str]

Threshold = 0.4

# Core inference function
def predict(texts: list[str]) -> list[float]:
    enc = model_loader.tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=70,
        return_tensors="pt"
    )
    enc = {k: v.to(DEVICE) for k, v in enc.items()}
    with torch.no_grad():
        probs = model_loader.model(**enc).logits.softmax(-1)[:, 1]
    return probs.cpu().tolist()

# Routes
@router.get("/")
def root():
    return {
        "status": "🟢 API live",
        "model": "bertweet-lora",
        "logging": "supabase" if SUPABASE_ENABLED else "disabled"
    }

@router.post("/check-text")
def check_text(payload: TextRequest):
    if not payload.texts:
        raise HTTPException(400, "No texts provided")

    results = []
    to_predict = []

    # Decide which texts go to model
    for text in payload.texts:
        if debug_rule_based_check(text):
            results.append({"blur": True, "score": 1.0})  # High confidence
        else:
            to_predict.append(text)
            results.append(None)  # placeholder

    # Predict only on remaining texts
    if to_predict:
        scores = predict(to_predict)
        idx = 0
        for i, r in enumerate(results):
            if r is None:
                score = scores[idx]
                results[i] = {"blur": score >= Threshold, "score": round(score, 4)}
                idx += 1

    return {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
        "results": results
    }

@router.post("/log-results")
def log_results(payload: TextRequest):
    if not SUPABASE_ENABLED:
        raise HTTPException(503, "Supabase logging is not configured")
    scores = predict(payload.texts)
    results = [{"blur": s >= Threshold, "score": round(s, 4)} for s in scores]
    success = sync_log_to_db(payload.texts, results)
    return {"logged": success}
