🧠 AI ATS Resume Screener

An intelligent Streamlit-based application that compares a Resume (PDF) with a Job Description (PDF) and generates:

✔ ATS Score (0–100)

✔ Matched & Missing Skills

✔ TF-IDF Similarity

✔ Sentence Transformer Similarity

✔ Named Entity Recognition (NER) skills

✔ Improvement suggestions

✔ Optional downloadable ATS Report
Report

🚀 Live App

📂 Features

Upload Resume (PDF)

Upload Job Description (PDF)

Extract text using PyPDF2

Clean & preprocess text

Skill extraction (regex + NER)

TF-IDF vector similarity (scikit-learn)

Semantic similarity (Sentence Transformers)

ATS Score calculation

Interactive UI built with Streamlit

🛠️ Installation
1️⃣ Clone the repository
git clone https://github.com/Laasyasree555/ATS-Resume-Screener.git
cd ATS-Resume-Screener

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Run the Streamlit app
streamlit run app.py

📦 Requirements

The app uses:

streamlit

sentence-transformers

pandas

numpy

scikit-learn

PyPDF2

spacy

python-docx

pyarrow

shap

plotly

(All included in requirements.txt)

📝 Project Structure
ATS-Resume-Screener/
│── app.py
│── requirements.txt
│── README.md
│── models/
│── results/
│── scripts/
│── data/

✨ Future Improvements

Add OCR for scanned PDFs

Integrate resume optimization tips

Expand skill keyword library

Add support for DOCX resume reading
