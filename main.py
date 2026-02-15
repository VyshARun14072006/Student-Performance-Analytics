import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

data = pd.read_csv("dataset/student_data.csv")

print("🎓 Student Performance Prediction (Console Version)")
print(data.head())

hours_studied = float(input("Hours Studied per Week: "))
attendance = float(input("Attendance %: "))
assignments_completed = float(input("Assignments Completed: "))

X_train = data[['Hours_Studied', 'Attendance', 'Assignments_Completed']]
y_train = data['Final_Score']

model = LinearRegression()
model.fit(X_train, y_train)

X_new = np.array([[hours_studied, attendance, assignments_completed]])
predicted_score = model.predict(X_new)[0]

print(f"\nPredicted Final Exam Score: {predicted_score:.2f}/100")
