from sentence_transformers import SentenceTransformer, util

# Load pre-trained sentence transformer
model = SentenceTransformer('all-MiniLM-L6-v2')

def semantic_similarity(resume_text, jd_text):
    # Encode both texts
    embeddings = model.encode([resume_text, jd_text], convert_to_tensor=True)
    # Compute cosine similarity
    similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
    return round(similarity * 100, 2)

# Example usage
if __name__ == "__main__":
    resume = """
    Developed REST APIs using FastAPI and integrated PostgreSQL database.
    Built CI/CD pipelines and handled JWT-based authentication.
    """
    jd = """
    Looking for a backend developer experienced in FastAPI, JWT, and PostgreSQL.
    """

    score = semantic_similarity(resume, jd)
    print(f"Semantic Similarity Score: {score}%")
