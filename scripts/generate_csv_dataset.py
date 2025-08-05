import pandas as pd
import re
from pathlib import Path

# File paths
RESUME_CSV = 'data/Resumes/UpdatedResumeDataSet.csv'
JD_CSV = 'data/Job_Description/dice_com-job_us_sample.csv'
OUTPUT_CSV = 'data/resume_match_dataset.csv'

# Simple text cleaner
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    return text

# Load and clean data
def load_and_prepare():
    resumes_df = pd.read_csv(RESUME_CSV)
    jds_df = pd.read_csv(JD_CSV)

    resumes_df['Resume'] = resumes_df['Resume'].apply(clean_text)
    jds_df['jobdescription'] = jds_df['jobdescription'].apply(clean_text)

    return resumes_df['Resume'], jds_df['jobdescription']

# Keyword overlap score
def calculate_match_score(resume, jd):
    resume_tokens = set(resume.split())
    jd_tokens = set(jd.split())
    overlap = resume_tokens.intersection(jd_tokens)
    return len(overlap)

# Generate labeled dataset
def generate_pair_dataset(resumes, job_descriptions, top_n_jds=100):
    data = []
    for resume_text in resumes[:100]:  # limit for speed
        for jd_text in job_descriptions[:top_n_jds]:
            score = calculate_match_score(resume_text, jd_text)
            label = 1 if score >= 15 else 0
            data.append({
                "resume_text": resume_text,
                "job_description": jd_text,
                "match_score": label
            })
    return pd.DataFrame(data)

def main():
    resumes, jds = load_and_prepare()
    df = generate_pair_dataset(resumes, jds)
    Path('data').mkdir(exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved dataset with {len(df)} pairs to {OUTPUT_CSV}")

if __name__ == '__main__':
    main()
