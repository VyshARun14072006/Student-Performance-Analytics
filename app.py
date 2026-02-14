import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Page configuration
st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓")

st.title("🎓 Student Performance Prediction System")
st.write("This ML-based system predicts a student's final exam score based on academic performance indicators.")

# Load dataset
data = pd.read_csv("../dataset/student_data.csv")


# Prepare features and target
X = data[['study_hours', 'attendance', 'assignments_avg', 'midterm_score']]
y = data['final_score']

# Train model
model = LinearRegression()
model.fit(X, y)

st.subheader("📊 Enter Student Details")

study_hours = st.slider("Study Hours per Day", 0, 12, 5)
attendance = st.slider("Attendance (%)", 0, 100, 75)
assignments_avg = st.slider("Assignments Average (%)", 0, 100, 70)
midterm_score = st.slider("Midterm Score (%)", 0, 100, 65)

if st.button("Predict Final Score"):
    input_data = np.array([[study_hours, attendance, assignments_avg, midterm_score]])
    prediction = model.predict(input_data)
    st.success(f"🎯 Predicted Final Score: {round(prediction[0], 2)}")

st.markdown("---")
st.write("Developed using Machine Learning (Linear Regression) with Streamlit Deployment")
