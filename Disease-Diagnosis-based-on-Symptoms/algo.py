import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import RandomForestClassifier

# Load dataset
data = pd.read_csv('C:\Users\Trupti\OneDrive\Desktop\NEW ML PROJECT\Disease-Diagnosis-based-on-Symptoms\datasets\Training.csv')

# Features (symptoms) and labels (disease)
X = data.iloc[:, :-1]
y = data['prognosis']

# Initialize models
nb_model = BernoulliNB()
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train models
nb_model.fit(X, y)
rf_model.fit(X, y)

# Predict using Naive Bayes
def predict_naive_bayes(symptoms):
    return nb_model.predict([symptoms])[0]

# Predict using Random Forest
def predict_random_forest(symptoms):
    return rf_model.predict([symptoms])[0]

# Compare both
def predict_with_all(symptoms):
    return {
        "NaiveBayes": predict_naive_bayes(symptoms),
        "RandomForest": predict_random_forest(symptoms)
    }

# Default predict (you can choose which model your app uses)
def predict(symptoms):
    # Example: use Random Forest as default
    return predict_random_forest(symptoms)
