import requests
import time

# ==== Your API Key Here ====
PERSPECTIVE_API_KEY = "AIzaSyBilVzOWXTcavRxk4W42wFKZ9UKHfTw2Mo"  # <-- Replace with your real key

# ==== API URL ====
PERSPECTIVE_API_URL = "https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze"

# ==== Main function to call Perspective API ====
def call_perspective_api(text, attributes=None, sleep_time=1):
    """
    Call Perspective API to analyze the text for hate speech attributes.
    
    Args:
        text (str): The text to analyze.
        attributes (list): List of attributes to request (default important ones).
        sleep_time (float): Seconds to sleep between calls to avoid throttling.
        
    Returns:
        dict: attribute name -> score
    """
    if attributes is None:
        attributes = ["TOXICITY", "SEVERE_TOXICITY", "INSULT", "THREAT", "IDENTITY_ATTACK", "PROFANITY"]

    headers = {"Content-Type": "application/json"}
    data = {
        "comment": {"text": text},
        "languages": ["en"],
        "requestedAttributes": {attr: {} for attr in attributes},
    }
    
    params = {"key": PERSPECTIVE_API_KEY}
    
    try:
        response = requests.post(PERSPECTIVE_API_URL, headers=headers, json=data, params=params)
        response.raise_for_status()
        result = response.json()
        scores = {attr: result["attributeScores"][attr]["summaryScore"]["value"] for attr in attributes}
        time.sleep(sleep_time)  # Sleep to be gentle with API
        return scores
    
    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")
        return {}

# ==== Example usage ====
if __name__ == "__main__":
    text_example = "I love to help people, It's so sweet."
    scores = call_perspective_api(text_example)
    
    print("Perspective API Scores:")
    for attribute, score in scores.items():
        print(f"{attribute}: {score:.3f}")
