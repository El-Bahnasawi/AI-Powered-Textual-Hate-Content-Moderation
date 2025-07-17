import re
from lexicons import (
    SWEAR_WORDS,
    HATE_KEYWORDS,
    HATE_HASHTAGS,
    HATE_SHORT_FORMS,
    HATE_EMOJIS
)
from test_cases import test_cases_1, test_cases_2

# Load offensive phrases from file
with open("en.txt", encoding="utf-8") as f:
    OFFENSIVE_PHRASES = set(line.strip().lower() for line in f if line.strip())

# -------------------- Preprocessing Utilities --------------------

def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[.]{2,}", " ", text)  # Normalize "..."
    return text

def clean_and_tokenize(text: str):
    text = normalize_text(text)
    return set(re.findall(r"\b\w+\b", text))

# -------------------- DEBUG Rule-Based Filter --------------------

def debug_rule_based_check(text: str) -> bool:
    text_norm = normalize_text(text)
    tokens = clean_and_tokenize(text_norm)

    if tokens & SWEAR_WORDS:
        print(f"🔍 Matched SWEAR_WORDS: {tokens & SWEAR_WORDS}")
        return True

    if tokens & HATE_KEYWORDS:
        print(f"🔍 Matched HATE_KEYWORDS: {tokens & HATE_KEYWORDS}")
        return True

    if tokens & HATE_SHORT_FORMS:
        print(f"🔍 Matched HATE_SHORT_FORMS: {tokens & HATE_SHORT_FORMS}")
        return True

    for phrase in OFFENSIVE_PHRASES:
        if phrase in text_norm:
            print(f"🔍 Matched OFFENSIVE_PHRASE: '{phrase}'")
            return True

    for tag in HATE_HASHTAGS:
        if tag in text_norm:
            print(f"🔍 Matched HATE_HASHTAG: {tag}")
            return True

    for emoji in HATE_EMOJIS:
        if emoji in text:
            print(f"🔍 Matched HATE_EMOJI: {emoji}")
            return True

    print("✅ No match found.")
    return False

# -------------------- Optional: Test Suite --------------------

if __name__ == "__main__":
    for text in test_cases_1 + test_cases_2:
        print(f"\n🧾 Text: {text}")
        result = debug_rule_based_check(text)
        print(f"Result: {'🔥 Flagged' if result else '✅ Safe'}")
