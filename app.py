import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load model
model = joblib.load('diabetes_model.pkl')

st.set_page_config(page_title="Diabetes Predictor", page_icon="🩺", layout="wide")
st.title("🩺 Diabetes Prediction App")
st.markdown("Enter health details below to predict the likelihood of diabetes.")

# Input form
with st.form("user_input"):
    st.subheader("🧾 Patient Information")

    col1, col2 = st.columns(2)

    pregnancies = col1.slider("Pregnancies", 0, 20, 1)
    glucose = col2.slider("Glucose", 0, 200, 120)
    blood_pressure = col1.slider("Blood Pressure", 0, 140, 70)
    skin_thickness = col2.slider("Skin Thickness", 0, 100, 20)
    insulin = col1.slider("Insulin", 0, 900, 80)
    bmi = col2.slider("BMI", 0.0, 70.0, 25.0)
    dpf = col1.slider("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
    age = col2.slider("Age", 10, 100, 33)

    submit = st.form_submit_button("🔍 Predict")

# Prediction
if submit:
    st.subheader("📈 Prediction Result")
    features = pd.DataFrame([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]],
                            columns=['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                                     'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'])

    result = model.predict(features)[0]
    proba = model.predict_proba(features)[0][result]

    if result == 1:
        st.error(f"🩺 Prediction: **Diabetic** ({proba * 100:.2f}% confidence)")
    else:
        st.success(f"🩺 Prediction: **Not Diabetic** ({proba * 100:.2f}% confidence)")

    # Feature importance chart
    st.subheader("📊 Feature Importance")
    feature_imp = model.feature_importances_
    labels = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
              'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

    fig, ax = plt.subplots()
    ax.barh(labels, feature_imp, color='skyblue')
    ax.set_xlabel("Importance Score")
    ax.set_title("Top Features Used by the Model")
    st.pyplot(fig)
