import streamlit as st
import pandas as pd
import joblib

# Load trained model and encoder
model = joblib.load("career_model.pkl")
encoder = joblib.load("label_encoder.pkl")

columns = ['Database Fundamentals','Computer Architecture','Distributed Computing Systems',
           'Cyber Security','Networking','Software Development','Programming Skills',
           'Project Management','Computer Forensics Fundamentals','Technical Communication',
           'AI ML','Software Engineering','Business Analysis','Communication skills',
           'Data Science','Troubleshooting skills','Graphics Designing']

st.title("Career Prediction App")

# Mapping text labels to numbers
mapping = {"Beginner":0, "Intermediate":1, "Professional":2, "Not Interested":3}

skills = []
for col in columns:
    level = st.selectbox(f"{col} level:", list(mapping.keys()))
    skills.append(mapping[level])

if st.button("Predict Career"):
    sample_input = pd.DataFrame([skills], columns=columns)
    prediction = model.predict(sample_input)
    career_name = encoder.inverse_transform(prediction)

    st.write("Available careers:", encoder.classes_)
    st.success(f"Suggested Career: {career_name[0]}")
