Diabetes Prediction App

This project is an introductory machine learning application designed to predict the likelihood of diabetes in patients using medical and lifestyle data. The app was built in Python with Streamlit to provide an interactive interface, making it easy for both technical and non-technical users to explore the dataset, understand the analysis, and view predictions in real time.

The dataset includes features such as Age, BMI, Blood Pressure, Cholesterol levels, Glucose levels (FBS and HBA1C), and other clinical indicators that are commonly associated with diabetes risk. To prepare the data, we performed data cleaning, normalization, and feature scaling using tools like Pandas, NumPy, and Scikit-learn.

For classification, two models were implemented:

K-Nearest Neighbors (KNN): A simple algorithm that classifies patients based on similarity to existing cases.

Artificial Neural Network (ANN): A beginner-level deep learning model built with TensorFlow/Keras to capture complex relationships in the data.

The app provides multiple functionalities:

Dataset Exploration: Users can view the raw and scaled datasets.

Exploratory Data Analysis (EDA): Basic visualizations using Seaborn and Matplotlib help users understand patterns and distributions.

Model Training and Prediction: Users can train models and predict diabetes outcomes based on patient data.

Performance Evaluation: Metrics such as accuracy, confusion matrix, ROC curves, and classification reports are displayed to evaluate model performance.

This project demonstrates how machine learning and data science techniques can be applied in healthcare to support early diagnosis and decision-making. While the models here are simplified for academic purposes, they showcase the process of preparing data, applying predictive algorithms, and deploying results in a user-friendly app.

By combining Python, machine learning, and Streamlit, this project highlights how students can bridge theory with practical implementation and gain hands-on experience in building end-to-end analytics solutions.
