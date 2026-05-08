import streamlit as st
from google import genai
import PyPDF2

# API client
client = genai.Client(api_key="YOUR_NEW_API_KEY")

st.title("📄 AI PDF Summarizer")

# 📂 Upload PDF
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    return text

if uploaded_file:
    st.success("PDF Uploaded Successfully ✅")

    # Extract text
    pdf_text = extract_text_from_pdf(uploaded_file)

    if st.button("Summarize PDF"):
        if pdf_text.strip():
            with st.spinner("Processing..."):
                response = client.models.generate_content(
                    model="gemini-flash-latest",
                    contents=f"Summarize this PDF in simple bullet points:\n{pdf_text[:8000]}"
                )
                st.subheader("📌 Summary:")
                st.write(response.text)
        else:
            st.error("Could not extract text from PDF")