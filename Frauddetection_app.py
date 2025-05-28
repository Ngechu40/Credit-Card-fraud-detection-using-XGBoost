import streamlit as st
import pandas as pd
import joblib
import shap
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Final selected features after preprocessing
FINAL_FEATURES = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
                  'V11', 'V12', 'V14', 'V16', 'V17', 'V18', 'V21', 'V27', 'V28', 'Hour']

# Load Dataset for Visualization
@st.cache_data
def load_data():
    df = pd.read_csv("creditcard.csv")
    df["Hour"] = (df["Time"] // 3600) % 24
    df.drop("Time", axis=1, inplace=True)
    return df

df = load_data()

# Load Trained Model
@st.cache_data
def load_model():
    return joblib.load("xgb_best_model.pkl")

model = load_model()

# App Title
st.title("Credit Card Fraud Detection Dashboard")

# Dataset Preview
st.subheader("Dataset Overview")
st.write(df.head())

# Fraud Class Distribution
st.subheader("Fraud vs. Non-Fraud Transactions")
fraud_counts = df["Class"].value_counts()
st.bar_chart(fraud_counts)

# Transaction Amount Distribution
st.subheader("Transaction Amount Distribution by Class")
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(df[df["Class"] == 0]["Amount"], bins=50, color='blue', alpha=0.6, label="Non-Fraudulent")
sns.histplot(df[df["Class"] == 1]["Amount"], bins=50, color='red', alpha=0.6, label="Fraudulent")
ax.set_xlabel("Transaction Amount ($)")
ax.set_ylabel("Count")
ax.set_title("Distribution of Transaction Amount")
plt.legend()
st.pyplot(fig)

# SHAP Feature Importance
st.subheader("SHAP Feature Importance for Fraud Detection")
X_shap = df[FINAL_FEATURES]
explainer = shap.Explainer(model, X_shap)
shap_values = explainer(X_shap)

fig, ax = plt.subplots(figsize=(10, 6))
shap.summary_plot(shap_values, X_shap, plot_type="bar", show=False)
st.pyplot(fig)

# Prediction Section
st.subheader("Fraud Prediction from Uploaded CSV")

uploaded_file = st.file_uploader("Upload CSV file for fraud prediction", type=["csv"])

if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)

    if 'Time' not in input_df.columns:
        st.error("Uploaded file must contain the 'Time' column to derive 'Hour'.")
    else:
        input_df["Hour"] = (input_df["Time"] // 3600) % 24
        input_df.drop("Time", axis=1, inplace=True)

        missing = [col for col in FINAL_FEATURES if col not in input_df.columns]
        if missing:
            st.error(f"Uploaded file is missing required features: {', '.join(missing)}")
        else:
            input_df = input_df[FINAL_FEATURES]

            predictions = model.predict(input_df)
            probabilities = model.predict_proba(input_df)[:, 1]

            result_df = input_df.copy()
            result_df["Fraud_Prediction"] = predictions
            result_df["Fraud_Probability"] = probabilities

            st.success("✅ Predictions completed successfully.")
            st.dataframe(result_df)

            csv = result_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Prediction Results", data=csv, file_name="fraud_predictions.csv", mime="text/csv")

# Footer
st.markdown("---")
st.caption("Credit Card Fraud Detection | Built with Streamlit & XGBoost")
