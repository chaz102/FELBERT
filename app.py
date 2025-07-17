from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline
from collections import Counter
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
import re
import os
import requests

# Load env vars
load_dotenv()

# Init app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Load FELBERT model for sentiment
sentiment = pipeline("text-classification", model="Chaz1003/FELBERT")

# Hugging Face Inference client
HF_TOKEN = os.environ.get("HF_TOKEN")
hf_client = InferenceClient(token=HF_TOKEN)

# Stopwords
stopwords = {
    "event", "the", "is", "at", "which", "on", "and", "a", "an", "but", "or", "to", "of", "in", "for", "with", "by", "it", "was",
    "as", "be", "are", "this", "that", "from", "so", "not", "no", "if", "we", "they", "you", "i", "me", "my", "do", "does",
    "did", "have", "has", "had", "will", "just", "about"
}


def translate_to_english(text):
    try:
        res = requests.get(f"http://localhost:5050/api/v1/auto/en/{text}")
        if res.ok:
            return res.json().get("translation", text)
        return text
    except Exception as e:
        print("Translation error:", e)
        return text


def tokenize_words(text):
    words = re.findall(r'\b\w+\b', text.lower())
    return [w for w in words if w not in stopwords]


def summarize_comments_phi4(comments):
    prompt = (
        "Summarize the following feedback and provide suggestions to improve the event.\n\n"
        "Please return your response in this format:\n"
        "Summary:\n<short summary>\n\n"
        "Suggestions:\n<bullet or short suggestions>\n\n"
    )
    prompt += "\n".join([f"- {c}" for c in comments])


    try:
        result = hf_client.chat_completion(
            model="microsoft/phi-4",
            messages=[{"role": "user", "content": prompt}]
        )
        return result.choices[0]["message"]["content"]
    except Exception as e:
        print("Summary error:", e)
        return "Summary unavailable."


@app.route("/batch-analyze", methods=["POST"])
def batch_analyze():
    comments = request.json.get("comments", [])
    results = []
    all_words = []

    for comment in comments:
        # 1. Translate for sentiment and summary
        translated = translate_to_english(comment)

        # 2. Analyze sentiment using translated version
        res = sentiment(translated)[0]
        label = "Negative" if res["label"] == "LABEL_0" else "Positive"
        score = round(res["score"] * 100, 2)

        results.append({
            "original": comment,
            "translated": translated,
            "sentiment": label,
            "confidence": score
        })

        # 3. Extract keywords from the ORIGINAL (untranslated) comment
        words = tokenize_words(comment)
        all_words.extend(words)

    keyword_counts = dict(Counter(all_words))
    summary = summarize_comments_phi4(comments)

    return jsonify({
        "results": results,
        "keywords": keyword_counts,
        "summary": summary
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
