import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import router
from model_loader import load_model

app = FastAPI()

# CORS for browser extension access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount all routes
app.include_router(router)

@app.on_event("startup")
def on_startup():
    print("🚀 Starting Hugging Face Space API")
    load_model()

    if os.getenv("SUPABASE_URL") and os.getenv("SUPABASE_KEY"):
        print("🔗 Supabase logging is enabled")
    else:
        print("⚠️ Supabase logging disabled (env not set)")

@app.on_event("shutdown")
def on_shutdown():
    print("🧹 Shutdown complete")
