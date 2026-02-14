import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
data = pd.read_csv("../dataset/student_data.csv")

print("Dataset Loaded Successfully!\n")
print(data.head())

# Features (Independent Variables)
X = data[['study_hours', 'attendance', 'assignments_avg', 'midterm_score']]

# Target (Dependent Variable)
y = data['final_score']

# Split the dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create Linear Regression model
model = LinearRegression()

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("\nModel Evaluation:")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Predict for a new student
new_student = np.array([[7, 85, 80, 78]])
predicted_score = model.predict(new_student)

print("\nPredicted Final Score for New Student:", predicted_score[0])
