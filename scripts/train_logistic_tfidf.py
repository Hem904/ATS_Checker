import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
from pathlib import Path

# === 1. Load Dataset ===
DATA_PATH = 'data/resume_match_dataset.csv'
df = pd.read_csv(DATA_PATH)

# === 2. Combine resume and JD as a single input ===
df['text_pair'] = df['resume_text'] + ' [SEP] ' + df['job_description']

# === 3. Split dataset ===
X_train, X_test, y_train, y_test = train_test_split(df['text_pair'], df['match_score'], test_size=0.2, random_state=42)

# === 4. TF-IDF Vectorization ===
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# === 5. Train Logistic Regression ===
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# === 6. Evaluate ===
y_pred = model.predict(X_test_vec)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("✅ Classification Report:")
print(classification_report(y_test, y_pred))

# === 7. Save Model and Vectorizer ===
Path("models").mkdir(exist_ok=True)
joblib.dump(model, "models/ats_match_model.pkl")
joblib.dump(vectorizer, "models/ats_tfidf_vectorizer.pkl")
print("✅ Model and vectorizer saved in /models")
