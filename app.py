from flask import Flask, request, jsonify
from transformers import pipeline
from dotenv import load_dotenv
import os, requests

app = Flask(__name__)

# Load pipelines
sentiment = pipeline("text-classification", model="Chaz1003/FELBERT")

load_dotenv()
LTR_URL = os.getenv("LTR_URL", "http://localhost:4000")

@app.route("/batch-analyze", methods=["POST"])
def batch_analyze():
    comments = request.json.get("comments", [])
    results = []
    translated_texts = []

    # Lazy load the summarizer
    summarizer = pipeline("summarization", model="google/pegasus-xsum")

    for comment in comments:
        lang = requests.post(f"{LTR_URL}/detect", json={"q": comment}).json()[0]["language"]

        if lang != "en":
            tr = requests.post(f"{LTR_URL}/translate", json={
                "q": comment,
                "source": lang,
                "target": "en"
            }).json()
            eng = tr["translatedText"]
        else:
            eng = comment

        translated_texts.append(eng)

        res = sentiment(eng)[0]
        label = "Negative" if res["label"] == "LABEL_0" else "Positive"
        score = round(res["score"] * 100, 2)

        results.append({
            "original": comment,
            "translated": eng,
            "sentiment": label,
            "confidence": score
        })

    long_text = " ".join(translated_texts)

    # Generate summary and suggestions
    summary = summarizer(long_text, max_length=100, min_length=40, do_sample=False)[0]["summary_text"]
    improve_prompt = "Based on the following feedback, suggest ways to improve the event: " + long_text
    improvements = summarizer(improve_prompt, max_length=100, min_length=40, do_sample=False)[0]["summary_text"]

    return jsonify({
        "results": results,
        "summary": summary,
        "suggestions": improvements
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
