from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load("best_sentiment_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    text = request.form["text"]

    text_tfidf = tfidf.transform([text])

    prediction = model.predict(text_tfidf)[0]

    if hasattr(model, "predict_proba"):
        confidence = max(model.predict_proba(text_tfidf)[0])
    else:
        confidence = None

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=round(float(confidence), 4)
        if confidence
        else "Not Available"
    )


if __name__ == "__main__":
    app.run(debug=True)