from transformers import pipeline
from flask import Flask, request, jsonify

app = Flask(__name__)
classifier = pipeline("text-classification", model="Chaz1003/FELBERT")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    user_input = data.get("text", "")
    result = classifier(user_input)
    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
