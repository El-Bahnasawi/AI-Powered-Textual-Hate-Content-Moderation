import streamlit as st
import requests
import time

# Constants
CHECK_API_URL = "https://medoxz543-hate-endpoint.hf.space/check-text"
LOG_API_URL = "https://medoxz543-hate-endpoint.hf.space/log-results"
HEADERS = {"Content-Type": "application/json"}

# App setup
st.set_page_config(page_title="Hate Speech Detector", page_icon="🧠", layout="wide")
st.title("🧠 Hate Speech Detector")

# Initialize session state
if "texts" not in st.session_state:
    st.session_state.texts = []
if "results" not in st.session_state:
    st.session_state.results = []

# Layout
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("📝 Input Texts")
    user_input = st.text_area(
        label="Enter one sentence per line:",
        height=400,
        placeholder="Example:\n- Jews are killers. We should kill them all\n- Have a nice day!"
    )
    detect_button = st.button("🚀 Run Detection")

# Detection and session update
if detect_button:
    if not user_input.strip():
        st.warning("⚠️ Please enter at least one sentence.")
    else:
        texts = [line.strip() for line in user_input.split("\n") if line.strip()]
        payload = {"texts": texts}
        start_time = time.time()

        try:
            response = requests.post(CHECK_API_URL, headers=HEADERS, json=payload)
            response.raise_for_status()
            data = response.json()
            elapsed = time.time() - start_time

            st.success(f"✅ Inference completed in {elapsed:.2f} seconds")
            st.markdown(f"🕒 Timestamp: `{data.get('timestamp', 'N/A')}`")

            st.session_state.texts = texts
            st.session_state.results = data["results"]

        except requests.exceptions.RequestException as e:
            st.error("❌ Failed to reach the API.")
            st.code(str(e), language="text")

# Results section
with col2:
    st.subheader("📊 Prediction Results")

    if st.session_state.texts and st.session_state.results:
        for idx, (text, result) in enumerate(zip(st.session_state.texts, st.session_state.results)):
            with st.container():
                st.markdown("---")
                st.markdown(f"**📝 Input #{idx+1}:** `{text}`")
                st.markdown(f"**🔎 Score:** `{round(result['score'], 3)}`")
                st.markdown(f"**📌 Prediction:** {'🔥 Hateful' if result['blur'] else '✅ Non-Hateful'}")

                disagree = st.button(f"❌ User Disagrees #{idx+1}", key=f"disagree_{idx}")
                if disagree:
                    try:
                        payload = {"texts": [text]}
                        response = requests.post(LOG_API_URL, headers=HEADERS, json=payload)
                        response.raise_for_status()
                        st.success("📥 Logged disagreement successfully.")
                    except Exception as e:
                        st.error(f"Logging failed: {e}")