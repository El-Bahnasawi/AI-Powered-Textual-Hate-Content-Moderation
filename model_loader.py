import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = None
model = None

def load_model():
    global tokenizer, model

    model_id = "medoxz543/hate-speech"
    print(f"🔄 Loading model from Hugging Face Hub: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id).to(DEVICE)
    model.eval()
    print("✅ Model ready")