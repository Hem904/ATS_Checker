import gradio as gr
from sentence_transformers import SentenceTransformer, util
import PyPDF2

# Load smaller model for Hugging Face (less memory)
model = SentenceTransformer('paraphrase-MiniLM-L3-v2')

def extract_text_from_pdf(pdf_file):
    reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def get_match_score(resume_pdf, job_description):
    try:
        resume_text = extract_text_from_pdf(resume_pdf)
        embeddings = model.encode([resume_text, job_description], convert_to_tensor=True)
        score = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
        return f"✅ Match Score: {round(score * 100, 2)}%"
    except Exception as e:
        return f"❌ Error: {str(e)}"

interface = gr.Interface(
    fn=get_match_score,
    inputs=[
        gr.File(label="Upload Resume (PDF)", file_types=[".pdf"]),
        gr.Textbox(lines=10, label="Paste Job Description")
    ],
    outputs="text",
    title="🧠 ATS Resume Match Checker",
    description="Upload your resume PDF and paste a job description to get an AI-powered semantic match score."
)

interface.launch()
