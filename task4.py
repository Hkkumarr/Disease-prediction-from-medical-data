import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ==========================
# Page Configuration
# ==========================

st.set_page_config(
    page_title="Disease Prediction System",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Disease Prediction from Medical Data")
st.write("Diabetes Prediction using Random Forest Classifier")

# ==========================
# Load Dataset
# ==========================

df = pd.read_csv("diabetes.csv")

# ==========================
# Dataset Preview
# ==========================

st.subheader("Dataset Preview")
st.dataframe(df.head())

# ==========================
# Features and Target
# ==========================

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# ==========================
# Train Test Split
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ==========================
# Train Model
# ==========================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# ==========================
# Model Evaluation
# ==========================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

st.subheader("Model Performance")
st.success(f"Accuracy: {accuracy*100:.2f}%")

# ==========================
# Confusion Matrix
# ==========================

st.subheader("Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots(figsize=(5, 4))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    ax=ax
)

ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix")

st.pyplot(fig)

# ==========================
# Feature Importance
# ==========================

st.subheader("Feature Importance")

importance = model.feature_importances_

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": importance
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

st.dataframe(feature_importance)

fig2, ax2 = plt.subplots(figsize=(8, 4))

ax2.bar(
    feature_importance["Feature"],
    feature_importance["Importance"]
)

plt.xticks(rotation=45)

st.pyplot(fig2)

# ==========================
# Patient Prediction
# ==========================

st.subheader("Predict New Patient")

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input(
        "Pregnancies",
        min_value=0,
        value=1
    )

    glucose = st.number_input(
        "Glucose",
        min_value=0,
        value=120
    )

    blood_pressure = st.number_input(
        "Blood Pressure",
        min_value=0,
        value=70
    )

    skin_thickness = st.number_input(
        "Skin Thickness",
        min_value=0,
        value=25
    )

with col2:
    insulin = st.number_input(
        "Insulin",
        min_value=0,
        value=100
    )

    bmi = st.number_input(
        "BMI",
        min_value=0.0,
        value=28.5
    )

    dpf = st.number_input(
        "Diabetes Pedigree Function",
        min_value=0.0,
        value=0.5
    )

    age = st.number_input(
        "Age",
        min_value=1,
        value=35
    )

if st.button("Predict Disease"):

    patient = pd.DataFrame(
        [[
            pregnancies,
            glucose,
            blood_pressure,
            skin_thickness,
            insulin,
            bmi,
            dpf,
            age
        ]],
        columns=X.columns
    )

    prediction = model.predict(patient)

    probability = model.predict_proba(patient)

    if prediction[0] == 1:
        st.error("⚠️ Diabetes Detected")
    else:
        st.success("✅ No Diabetes Detected")

    st.write(
        f"Diabetes Probability: {probability[0][1]*100:.2f}%"
    )

# ==========================
# Classification Report
# ==========================

st.subheader("Classification Report")

report = classification_report(
    y_test,
    y_pred,
    output_dict=True
)

report_df = pd.DataFrame(report).transpose()

st.dataframe(report_df)