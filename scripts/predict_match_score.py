import joblib

# === Load Saved Model and Vectorizer ===
model = joblib.load("models/ats_match_model.pkl")
vectorizer = joblib.load("models/ats_tfidf_vectorizer.pkl")

# === Define Prediction Function ===
def predict_match_score(resume_text, job_description):
    input_text = resume_text.lower() + ' [SEP] ' + job_description.lower()
    input_vec = vectorizer.transform([input_text])
    prediction = model.predict(input_vec)[0]
    probability = model.predict_proba(input_vec)[0][1]  # Probability for label 1
    return prediction, round(probability * 100, 2)

if __name__ == "__main__":
    resume = """
    Developed REST APIs using FastAPI and integrated PostgreSQL database.
    Built CI/CD pipelines and handled JWT-based authentication.
    """
    jd = """
    Looking for a backend developer experienced in FastAPI, JWT, and PostgreSQL.
    """

    label, confidence = predict_match_score(resume, jd)
    print(f"Prediction: {'✅ Match' if label else '❌ Not a Match'}")
    print(f"Confidence Score: {confidence}%")

