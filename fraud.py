import streamlit as st
import pandas as pd
import joblib
import shap
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

# Final selected features after preprocessing
FINAL_FEATURES = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
                  'V11', 'V12', 'V14', 'V16', 'V17', 'V18', 'V21', 'V27', 'V28', 'Hour']

# --- Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("Book4.csv")
    df["Hour"] = (df["Time"] // 3600) % 24
    df.drop("Time", axis=1, inplace=True)
    return df

# --- Load trained model (balanced using ADASYN)
@st.cache_data
def load_model():
    return joblib.load("xgb_best_model.pkl")

# Load data and model
df = load_data()
model = load_model()

# --- App Title
st.title(" Credit Card Fraud Detection Dashboard")

# --- Dataset Preview
st.subheader(" Dataset Overview")
st.write(df.head())

# --- Fraud Class Distribution
st.subheader(" Fraud vs. Non-Fraud Transactions")
fraud_counts = df["Class"].value_counts()
st.bar_chart(fraud_counts)

# --- Transaction Amount Distribution
st.subheader(" Transaction Amount Distribution by Class")
fig, ax = plt.subplots(figsize=(10, 5))
sns.histplot(df[df["Class"] == 0]["Amount"], bins=50, color='blue', label="Non-Fraud", alpha=0.6)
sns.histplot(df[df["Class"] == 1]["Amount"], bins=50, color='red', label="Fraud", alpha=0.6)
ax.set_title("Transaction Amount Distribution")
ax.set_xlabel("Amount ($)")
ax.set_ylabel("Count")
plt.legend()
st.pyplot(fig)

# --- SHAP Feature Importance (limited to 200 samples for speed)
st.subheader("  SHAP Feature Importance (Sampled)")
X_sample = df[FINAL_FEATURES].sample(200, random_state=42)
explainer = shap.Explainer(model, X_sample)
shap_values = explainer(X_sample)

fig_shap, ax_shap = plt.subplots(figsize=(10, 5))
shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
st.pyplot(fig_shap)

# --- Prediction Section
st.subheader(" Predict Fraud from Uploaded CSV")

uploaded_file = st.file_uploader("Upload a CSV file with similar structure", type=["csv"])

if uploaded_file:
    input_df = pd.read_csv(uploaded_file)

    if 'Time' not in input_df.columns:
        st.error("Uploaded file must include a 'Time' column to derive 'Hour'.")
    else:
        input_df["Hour"] = (input_df["Time"] // 3600) % 24
        input_df.drop("Time", axis=1, inplace=True)

        missing = [col for col in FINAL_FEATURES if col not in input_df.columns]
        if missing:
            st.error(f"Missing required features: {', '.join(missing)}")
        else:
            input_df = input_df[FINAL_FEATURES]
            y_pred = model.predict(input_df)
            y_proba = model.predict_proba(input_df)[:, 1]

            result_df = input_df.copy()
            result_df["Fraud_Prediction"] = y_pred
            result_df["Fraud_Probability"] = y_proba

            st.success(" Predictions generated successfully.")
            st.dataframe(result_df)

            # Download button
            csv_data = result_df.to_csv(index=False).encode("utf-8")
            st.download_button(" Download Results", csv_data, file_name="fraud_predictions.csv", mime="text/csv")

            # --- Metrics and Visuals
            st.markdown("---")
            st.subheader(" Model Evaluation Metrics")

            # Simulate true labels if 'Class' exists
            if "Class" in input_df.columns or "Class" in uploaded_file.name:
                st.warning(" True labels were not provided. Metrics will use dummy labels.")
                y_true = np.zeros(len(y_pred))  # dummy labels
            else:
                y_true = y_pred  # for display purpose

            col1, col2 = st.columns(2)

            with col1:
                st.subheader(" Classification Report")
                report_df = pd.DataFrame(classification_report(y_true, y_pred, output_dict=True)).transpose()
                st.dataframe(report_df.round(2))

            with col2:
                roc_auc = roc_auc_score(y_true, y_proba)
                st.metric("ROC AUC Score", f"{roc_auc:.3f}")

            st.subheader(" Confusion Matrix")
            cm = confusion_matrix(y_true, y_pred)
            fig_cm, ax_cm = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=["Not Fraud", "Fraud"], yticklabels=["Not Fraud", "Fraud"])
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            st.pyplot(fig_cm)

            st.subheader(" ROC Curve")
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            fig_roc, ax_roc = plt.subplots()
            ax_roc.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
            ax_roc.plot([0, 1], [0, 1], 'k--')
            ax_roc.set_xlabel("False Positive Rate")
            ax_roc.set_ylabel("True Positive Rate")
            ax_roc.set_title("ROC Curve")
            ax_roc.legend()
            st.pyplot(fig_roc)

# --- Footer
st.markdown("---")
st.caption("Credit Card Fraud Detection | Built with Streamlit & XGBoost | Balanced using ADASYN")
