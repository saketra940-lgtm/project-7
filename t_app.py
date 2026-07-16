from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# ==========================
# Load Saved Model & Vectorizer
# ==========================

model = joblib.load("logistic_regression_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")


# ==========================
# Home Page
# ==========================

@app.route("/")
def home():

    return render_template(
        "index.html",
        prediction_text=None
    )


# ==========================
# Prediction
# ==========================

@app.route("/predict", methods=["POST"])
def predict():

    tweet = request.form["tweet"]

    if tweet.strip() == "":

        return render_template(
            "index.html",
            error="Please enter a tweet."
        )

    # TF-IDF Transformation
    tweet_vector = tfidf.transform([tweet])

    # Prediction
    prediction = model.predict(tweet_vector)[0]

    # Probability
    probability = model.predict_proba(tweet_vector)[0]

    disaster_prob = round(probability[1] * 100, 2)
    non_disaster_prob = round(probability[0] * 100, 2)

    confidence = round(max(probability) * 100, 2)

    # Statistics
    word_count = len(tweet.split())

    character_count = len(tweet)

    if prediction == 1:

        result = "🚨 Disaster Tweet"

        result_class = "danger"

    else:

        result = "✅ Non-Disaster Tweet"

        result_class = "success"

    return render_template(

        "index.html",

        tweet=tweet,

        prediction_text=result,

        result_class=result_class,

        confidence=confidence,

        disaster_prob=disaster_prob,

        non_disaster_prob=non_disaster_prob,

        word_count=word_count,

        character_count=character_count,

        model_name="Logistic Regression",

        vectorizer="TF-IDF",

        dataset="10,000 Disaster Tweets"

    )


# ==========================

if __name__ == "__main__":

    app.run(debug=True)