from transformers import pipeline
from flask import Flask, request, jsonify

app = Flask(__name__)
classifier = pipeline("text-classification", model="Chaz1003/FELBERT")

@app.route("/", methods=["GET"])
def home():
    return "Sentiment API is running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    user_input = data.get("text", "")
    
    # Run model prediction
    result = classifier(user_input)[0]
    label = result["label"]
    score = round(result["score"] * 100, 2)

    sentiment = "Negative" if label == "LABEL_0" else "Positive"

    # Format for Dialogflow
    return jsonify({
        "fulfillmentText": f"{sentiment} with {score}% confidence"
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)