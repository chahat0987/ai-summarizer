import streamlit as st
from google import genai
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
from dotenv import load_dotenv
load_dotenv()

client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])

st.title("🤖 PDF RAG Chatbot")

# Upload PDF
uploaded_file = st.file_uploader("Upload PDF", type="pdf")

def extract_text(file):
    reader = PyPDF2.PdfReader(file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

def split_text(text, chunk_size=500):
    chunks = []
    for i in range(0, len(text), chunk_size):
        chunks.append(text[i:i+chunk_size])
    return chunks

if uploaded_file:
    st.success("PDF Uploaded ✅")

    # Extract + chunk
    text = extract_text(uploaded_file)
    chunks = split_text(text)

    # Vectorize
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(chunks)

    query = st.text_input("Ask something from PDF:")

    if st.button("Ask"):
        if query:
            # Convert query → vector
            query_vec = vectorizer.transform([query])

            # Similarity
            scores = cosine_similarity(query_vec, vectors)[0]

            # Top 3 chunks
            top_indices = scores.argsort()[-3:][::-1]
            context = "\n".join([chunks[i] for i in top_indices])

            # Gemini answer
            prompt = f"""
            Answer the question using only this context:

            {context}

            Question: {query}
            """

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )

            st.subheader("💡 Answer:")
            st.write(response.text)