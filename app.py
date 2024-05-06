import streamlit as st
import pandas as pd
import time
import PyPDF2

# Importing the predict function from model.py
from model import predict 

st.title("Job Recommendation System")

uploaded_file = st.file_uploader("Upload your Resume", type='pdf')

if uploaded_file is not None:
    print(uploaded_file)

text = "We believe these 5 jobs are a good fit for you!"

def stream_data():
    pdf_reader = PyPDF2.PdfReader(uploaded_file)
    resume_text = ""

    for page in pdf_reader.pages:
        resume_text += page.extract_text()

    result = predict(resume_text)

    for word in text.split(" "):
        yield word + " "
        time.sleep(0.02)

    yield pd.DataFrame(result)

if st.button("Recommend me Jobs"):
    st.write_stream(stream_data())

st.divider()

st.caption("Developed and packaged by:")
st.caption("1. Kedar Dhamankar")
st.caption("2. Saumya Karia")
st.caption("3. Rucha Patil")
st.caption("4. Dhruv Chugh")
st.caption("5. Aryan Pawaskar")