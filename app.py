from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline
import os

app = Flask(__name__)
CORS(app)

# Load sentiment model
sentiment = pipeline("text-classification", model="Chaz1003/FELBERT")

@app.route("/batch-analyze", methods=["POST"])
def batch_analyze():
    comments = request.json.get("comments", [])
    results = []

    for comment in comments:
        res = sentiment(comment)[0]
        label = "Negative" if res["label"] == "LABEL_0" else "Positive"
        score = round(res["score"] * 100, 2)

        results.append({
            "original": comment,
            "sentiment": label,
            "confidence": score
        })

    return jsonify({"results": results})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
