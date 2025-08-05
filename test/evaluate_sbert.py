import pandas as pd
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics import roc_auc_score, classification_report

# Load data
df = pd.read_csv('data/resume_match_dataset.csv')
df = df.sample(200, random_state=42)  # small subset for quick test

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Get similarity scores
scores = []
for _, row in df.iterrows():
    emb = model.encode([row['resume_text'], row['job_description']], convert_to_tensor=True)
    similarity = util.pytorch_cos_sim(emb[0], emb[1]).item()
    scores.append(similarity)

df['similarity'] = scores

# Binary predictions with threshold
threshold = 0.70
df['predicted'] = df['similarity'] > threshold

# Evaluation
print(f"ROC AUC Score: {roc_auc_score(df['match_score'], df['similarity'])}")
print(classification_report(df['match_score'], df['predicted']))
