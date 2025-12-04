# app.py
# AI-Powered ATS Resume Screening — Full version (Streamlit)
# Paste this entire file into project/app/app.py

import streamlit as st
from PyPDF2 import PdfReader
import re
import tempfile
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer, util
import plotly.express as px
import spacy
from html import escape
from datetime import datetime
import base64

# -------------------- Utility / Helpers --------------------
@st.cache_data
def load_sbert_model(model_name="all-mpnet-base-v2"):
    # uses sentence-transformers model; cached so not reloaded each run
    return SentenceTransformer(model_name)

@st.cache_data
def load_spacy_model():
    # spaCy model (en_core_web_sm)
    return spacy.load("en_core_web_sm")

def extract_text_from_pdf_bytes(pdf_bytes):
    try:
        # PdfReader accepts bytes-like object from uploaded_file.read()
        reader = PdfReader(pdf_bytes)
        text = ""
        for p in reader.pages:
            page_text = p.extract_text()
            if page_text:
                text += page_text + " "
        return text
    except Exception as e:
        return ""

def clean_text(text):
    if not text:
        return ""
    # basic cleaning and normalization
    text = re.sub(r"\r\n|\r|\n", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text

def extract_skills_from_text(text, skills_list):
    text_low = text.lower()
    found = [s for s in skills_list if s.lower() in text_low]
    return sorted(found)

def highlight_text_html(text, matched, missing):
    # simple HTML-based highlighting (case-insensitive)
    safe = escape(text)
    # order: highlight longer phrases first to avoid partial overlaps
    phrases = sorted(matched + missing, key=lambda s: -len(s))
    for ph in phrases:
        ph_safe = escape(ph)
        # case-insensitive replacement using regex
        if ph in matched:
            color = "#b6f5c7"  # light green
        else:
            color = "#ffd7d7"  # light red
        safe = re.sub(r"(?i)\b" + re.escape(ph_safe) + r"\b",
                      f"<mark style='background:{color}; padding:0.1rem;'>{ph_safe}</mark>",
                      safe)
    # allow basic HTML
    return safe.replace("\n", "<br>")

def get_download_link(text, filename="ats_report.txt"):
    b64 = base64.b64encode(text.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">⬇️ Download ATS report</a>'
    return href

# -------------------- Skills master (customize) --------------------
SKILLS_MASTER = [
    # analytics/business
    "analytics","data analysis","dashboard","dashboards","trends",
    "root cause analysis","rca","a/b testing","experiments","growth experiments",
    "retention","user repeat","pricing","sla","loyalty programs","ux","search","segmentation",
    "growth","hyperlocal","campaigns","engagement","strategy","execution","cross-functional",
    "implementation","program management",
    # soft skills
    "communication","creative","analytical","problem solving","collaboration","teamwork",
    "ownership","curiosity","innovation","decision-making",
    # tools
    "excel","sql","tableau","power bi","powerbi","data visualization","python","pandas","numpy",
    "sql","sql server","bigquery"
]

# -------------------- Page layout --------------------
st.set_page_config(page_title="AI ATS Resume Screener", layout="wide")
st.title("📄 AI Resume Screener — Resume vs Job Description")
st.write("Upload a Resume (PDF) and a Job Description (PDF). The app returns ATS score, matched/missing skills, similarity metrics, NER, and improvement suggestions.")

# Sidebar: model selection and options
with st.sidebar:
    st.header("Settings")
    model_choice = st.selectbox("Embedding model (SBERT)", ["all-mpnet-base-v2", "all-MiniLM-L6-v2"], index=0)
    show_raw = st.checkbox("Show raw cleaned text", value=False)
    enable_ner = st.checkbox("Enable NER extraction (spaCy)", value=True)
    export_pdf = st.checkbox("Enable download report", value=True)
    st.markdown("---")
    st.caption("Note: Bigger models are slower but more accurate. The first load downloads the model and may take 30–60s.")

# -------------------- Upload files --------------------
col1, col2 = st.columns(2)
with col1:
    resume_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"], key="resume")
with col2:
    jd_file = st.file_uploader("Upload Job Description (PDF)", type=["pdf"], key="jd")

if resume_file and jd_file:
    # Read bytes
    resume_bytes = resume_file.read()
    jd_bytes = jd_file.read()

    # Extract and clean text
    resume_raw = extract_text_from_pdf_bytes(resume_bytes)
    jd_raw = extract_text_from_pdf_bytes(jd_bytes)

    if not resume_raw:
        st.error("Could not extract text from the resume PDF. If it's a scanned image, use OCR or paste text manually.")
    if not jd_raw:
        st.error("Could not extract text from the JD PDF. Try re-uploading.")

    resume_clean = clean_text(resume_raw)
    jd_clean = clean_text(jd_raw)

    if show_raw:
        st.subheader("Cleaned Resume Text")
        st.write(resume_clean[:4000] + ("..." if len(resume_clean) > 4000 else ""))
        st.subheader("Cleaned JD Text")
        st.write(jd_clean[:4000] + ("..." if len(jd_clean) > 4000 else ""))

    # Skill extraction
    resume_skills = extract_skills_from_text(resume_clean, SKILLS_MASTER)
    jd_skills = extract_skills_from_text(jd_clean, SKILLS_MASTER)

    matched = sorted(list(set(resume_skills).intersection(set(jd_skills))))
    missing = sorted(list(set(jd_skills).difference(set(resume_skills))))

    # TF-IDF similarity
    try:
        tfidf = TfidfVectorizer(stop_words="english")
        vecs = tfidf.fit_transform([resume_clean, jd_clean])
        tfidf_sim = float((vecs * vecs.T).toarray()[0,1])
    except Exception:
        tfidf_sim = 0.0

    # Load SBERT model
    with st.spinner("Loading embedding model (this may take a while first time)..."):
        model = load_sbert_model(model_choice)

    emb_resume = model.encode(resume_clean, convert_to_tensor=True)
    emb_jd = model.encode(jd_clean, convert_to_tensor=True)
    emb_sim = float(util.pytorch_cos_sim(emb_resume, emb_jd).item())

    # Skill score
    skill_score = (len(matched) / len(jd_skills)) if len(jd_skills) > 0 else 0.0

    # Final ATS Score — weights can be tuned
    final_score = round((skill_score * 0.4 + tfidf_sim * 0.3 + emb_sim * 0.3) * 100, 2)

    # -------------------- Display results --------------------
    st.subheader("📊 ATS Results")
    c1, c2, c3, c4 = st.columns([1,1,1,1])
    c1.metric("Final ATS Score", f"{final_score} / 100")
    c2.metric("Skill Match (%)", f"{round(skill_score*100,2)}%")
    c3.metric("TF-IDF Similarity", round(tfidf_sim,4))
    c4.metric("Embedding Similarity", round(emb_sim,4))

    # Matched / Missing lists
    st.write("### ✅ Matched Skills")
    st.write(matched if matched else "None detected")

    st.write("### ❌ Missing JD Skills (recommended to add)")
    st.write(missing if missing else "None — good match!")

    # Highlighted resume display
    st.write("### ✨ Resume Preview (highlighted)")
    highlighted = highlight_text_html(resume_raw, matched, missing)
    st.markdown(highlighted, unsafe_allow_html=True)

    # Charts
    st.write("### 📈 Visualizations")
    fig = px.pie(names=["Matched", "Missing"], values=[len(matched), len(missing)],
                 title="Matched vs Missing Skills")
    st.plotly_chart(fig, use_container_width=True)

    score_comp = px.bar(
        x=["Skill(%)","TF-IDF","Embedding"],
        y=[round(skill_score*100,2), round(tfidf_sim*100,2), round(emb_sim*100,2)],
        labels={"x":"Metric", "y":"Percentage"},
        title="Score component comparison"
    )
    st.plotly_chart(score_comp, use_container_width=True)

    # NER
    if enable_ner:
        try:
            nlp = load_spacy_model()
            doc = nlp(resume_raw[:5000])  # limit for speed
            entities = [(ent.text, ent.label_) for ent in doc.ents]
            st.write("### 🧾 Extracted Entities (spaCy)")
            st.write(entities[:100])
        except Exception as e:
            st.warning("spaCy NER failed or model not installed. Run `python -m spacy download en_core_web_sm` in your environment.")

    # Improvement suggestions
    st.write("### 🛠️ Resume Improvement Suggestions")
    if missing:
        for s in missing:
            st.write(f"- Add or highlight experience / project / coursework related to **{s}**")
    else:
        st.write("No missing skills identified. Good match!")

    # Job fit summary
    st.write("### 🔎 Job Fit Summary")
    fit_text = (
        f"This resume best fits the JD with a final ATS score of {final_score}. "
        f"Matched skills: {', '.join(matched) if matched else 'None'}. "
        f"Key recommended additions: {', '.join(missing) if missing else 'None'}."
    )
    st.info(fit_text)

    # Downloadable report
    if export_pdf:
        report_text = (
            f"ATS Report Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S (UTC)')}\n\n"
            f"JD File: {jd_file.name}\nResume File: {resume_file.name}\n\n"
            f"Final ATS Score: {final_score}/100\nTF-IDF: {tfidf_sim:.4f}\nEmbedding: {emb_sim:.4f}\n\n"
            f"Matched Skills: {', '.join(matched)}\nMissing Skills: {', '.join(missing)}\n\n"
            "Recommendations:\n" + "\n".join([f"- Add experience related to {s}" for s in missing])
        )
        st.markdown(get_download_link(report_text, filename="ats_report.txt"), unsafe_allow_html=True)

else:
    st.info("Upload both Resume and JD (PDF). The app will compute ATS scores and suggestions.")

# footer
st.markdown("---")
st.caption("Built with Streamlit • SBERT embeddings • spaCy NER • TF-IDF. Keep sensitive personal data private.")
