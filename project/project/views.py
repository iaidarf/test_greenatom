from django.shortcuts import render
import joblib
from pathlib import Path


def index(request):
    return render(request, "base.html")

def prediction(request):
    BASE_DIR = Path(__file__).resolve().parent.parent

    vectorizer_pos_neg = joblib.load(BASE_DIR / "project/vectorizer_pos_neg.joblib")
    vectorizer_rating = joblib.load(BASE_DIR / "project/vectorizer_rating.joblib")

    prediction_pos_neg = joblib.load(BASE_DIR / "project/model_pos_neg.joblib")
    prediction_rating = joblib.load(BASE_DIR / "project/model_rating.joblib")

    review = request.POST["review"]

    review_pos_neg = vectorizer_pos_neg.transform([review])
    review_rating = vectorizer_rating.transform([review])

    pos_neg_ = prediction_pos_neg.predict(review_pos_neg)[0]
    pos_neg = 'Positive' if (pos_neg_ == 1) else 'Negative'
    rating = prediction_rating.predict(review_rating)[0]

    settings = {"review": review}

    return render(request, "base.html", {"review_text": review, "pos_neg": pos_neg, "rating":rating, "settings": settings})
