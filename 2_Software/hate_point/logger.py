import os
from supabase import create_client, Client

SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

def sync_log_to_db(texts, results):
    data = [
        {"text": t, "blur": r["blur"], "score": r["score"]}
        for t, r in zip(texts, results)
    ]
    try:
        response = client.table("cases").insert(data).execute()
        if response.data:
            print(f"✅ Logged {len(response.data)} rows")
            return True
    except Exception as e:
        print("❌ Supabase logging failed:", e)
    return False
