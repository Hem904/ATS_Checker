from flask import Flask, request, jsonify,render_template
from sentence_transformers import SentenceTransformer, util
import PyPDF2
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)
model = SentenceTransformer('all-MiniLM-L6-v2')



def extract_text_from_pdf(file):
    pdf = PyPDF2.PdfReader(file)
    text = ''
    for page in pdf.pages:
        text += page.extract_text() or ''
    return text

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/api/match_score', methods=['POST'])
def match_score():
    if 'resume_file' not in request.files or 'job_description' not in request.form:
        return jsonify({"error": "Missing resume_file or job_description"}), 400

    resume_file = request.files['resume_file']
    job_description = request.form['job_description']

    if not resume_file.filename.lower().endswith('.pdf'):
        return jsonify({"error": "Only PDF resumes allowed"}), 400

    resume_text = extract_text_from_pdf(resume_file)

    embeddings = model.encode([resume_text, job_description], convert_to_tensor=True)
    score = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
    percent = round(score * 100, 2)

    return jsonify({
        "match_score": percent,
        "status": "success"
    })

if __name__ == '__main__':
    app.run(debug=True)
