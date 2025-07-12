from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline
import os
from collections import Counter
import re

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

sentiment = pipeline("text-classification", model="Chaz1003/FELBERT")

def tokenize_words(text):
    return re.findall(r'\b\w+\b', text.lower())

@app.route("/batch-analyze", methods=["POST"])
def batch_analyze():
    comments = request.json.get("comments", [])
    results = []
    all_words = []

    for comment in comments:
        res = sentiment(comment)[0]
        label = "Negative" if res["label"] == "LABEL_0" else "Positive"
        score = round(res["score"] * 100, 2)

        results.append({
            "comment": comment,
            "sentiment": label,
            "confidence": score
        })

        words = tokenize_words(comment)
        all_words.extend(words)

    keyword_counts = dict(Counter(all_words))

    return jsonify({
        "results": results,
        "keywords": keyword_counts
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
