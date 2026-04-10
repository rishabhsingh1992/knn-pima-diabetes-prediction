import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix

df = pd.read_csv("../../Datasets/diabetes.csv")

X = df.drop(columns=["Outcome"])
y = df["Outcome"]

(X_train, X_test, y_train, y_test) = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()

# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

model = KNeighborsClassifier(n_neighbors=5)
# model.fit(X_train_scaled, y_train)

# y_pred = model.predict(X_test_scaled)

model_pipeline = Pipeline([
    ("scaler", scaler),
    ("model", model)
])

model_pipeline.fit(X_train, y_train)

y_pred = model_pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

# print(f"Accuracy: {accuracy * 100:.2f}%")
# print(confusion_matrix(y_test, y_pred))

joblib.dump(model_pipeline, "models/knn_diabetes_model_pipeline.pkl")
# joblib.dump(scaler, "models/knn_diabetes_scaler.pkl")
