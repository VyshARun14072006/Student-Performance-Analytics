import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Title
st.title("🎓 Student Performance Prediction System")

# Load dataset
data = pd.read_csv("dataset/student_data.csv")
st.subheader("Sample Dataset")
st.write(data.head())

# Input sliders for features
st.subheader("Enter Student Details")
hours_studied = st.slider("Hours Studied per Week", 0, 40, 10)
attendance = st.slider("Attendance %", 0, 100, 80)
assignments_completed = st.slider("Assignments Completed", 0, 20, 15)

# Predict button
if st.button("Predict Final Score"):
    X_train = data[['Hours_Studied', 'Attendance', 'Assignments_Completed']]
    y_train = data['Final_Score']
    
    model = LinearRegression()
    model.fit(X_train, y_train)

    X_new = np.array([[hours_studied, attendance, assignments_completed]])
    predicted_score = model.predict(X_new)[0]

    st.success(f"Predicted Final Exam Score: {predicted_score:.2f}/100")
