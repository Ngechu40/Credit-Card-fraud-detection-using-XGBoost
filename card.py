import streamlit as st
import pandas as pd
import joblib
import shap
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Final selected features after preprocessing
FINAL_FEATURES = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
                  'V11', 'V12', 'V14', 'V16', 'V17', 'V18', 'V21', 'V27', 'V28', 'Hour']

# Cache dataset loading with optional sampling
@st.cache_data
def load_data(sample_size=None):
    df = pd.read_csv("creditcard.csv")
    df["Hour"] = (df["Time"] // 3600) % 24
    df.drop("Time", axis=1, inplace=True)
    if sample_size:
        df = df.sample(sample_size, random_state=42) 
    return df

# Cache model loading
@st.cache_data
def load_model():
    return joblib.load("xgb_best_model.pkl")

# Cache for plots
@st.cache_data
def plot_amount_distribution(data):
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data[data["Class"] == 0]["Amount"], bins=50, color='blue', alpha=0.6, label="Non-Fraudulent")
    sns.histplot(data[data["Class"] == 1]["Amount"], bins=50, color='red', alpha=0.6, label="Fraudulent")
    ax.set_xlabel("Transaction Amount ($)")
    ax.set_ylabel("Count")
    ax.set_title("Distribution of Transaction Amount")
    plt.legend()
    return fig

# Batch prediction for large datasets
def batch_predict(model, data, batch_size=1000):
    predictions = []
    probabilities = []
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        predictions.extend(model.predict(batch))
        probabilities.extend(model.predict_proba(batch)[:, 1])
    return np.array(predictions), np.array(probabilities)

# Load dataset and model
df = load_data(sample_size=10000)  
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
fig = plot_amount_distribution(df)
st.pyplot(fig)

# SHAP Feature Importance
st.subheader("SHAP Feature Importance for Fraud Detection")
X_shap_sampled = df[FINAL_FEATURES].sample(1000, random_state=42)  
explainer = shap.Explainer(model, X_shap_sampled)  
shap_values = explainer(X_shap_sampled)

fig, ax = plt.subplots(figsize=(10, 6))
shap.summary_plot(shap_values, X_shap_sampled, plot_type="bar", show=False)
st.pyplot(fig)

# Evaluation Metrics
st.subheader("Model Evaluation Metrics")
sampled_df = df.sample(2000, random_state=42)  # Use a subset of rows for evaluation
X_eval = sampled_df[FINAL_FEATURES]
y_eval_true = sampled_df["Class"]
y_eval_pred = model.predict(X_eval)

# Compute metrics
accuracy = accuracy_score(y_eval_true, y_eval_pred)
precision = precision_score(y_eval_true, y_eval_pred)
recall = recall_score(y_eval_true, y_eval_pred)
f1 = f1_score(y_eval_true, y_eval_pred)
conf_matrix = confusion_matrix(y_eval_true, y_eval_pred)

# Display metrics
st.write("**Accuracy:**", accuracy)
st.write("**Precision:**", precision)
st.write("**Recall:**", recall)
st.write("**F1-Score:**", f1)

st.write("**Confusion Matrix:**")
st.write(conf_matrix)

# Prediction Section
st.subheader("Fraud Prediction from Uploaded CSV")

uploaded_file = st.file_uploader("Upload CSV file for fraud prediction", type=["csv"])

if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)

    # Validate required columns
    missing = [col for col in ["Time"] + FINAL_FEATURES if col not in input_df.columns]
    if missing:
        st.error(f"Uploaded file is missing required features: {', '.join(missing)}")
    else:
        # Transform input data
        input_df["Hour"] = (input_df["Time"] // 3600) % 24
        input_df.drop("Time", axis=1, inplace=True)
        input_df = input_df[FINAL_FEATURES]

        # Batch predictions
        predictions, probabilities = batch_predict(model, input_df)

        # Combine results
        result_df = input_df.copy()
        result_df["Fraud_Prediction"] = predictions
        result_df["Fraud_Probability"] = probabilities

        st.success("Predictions completed successfully.")
        st.dataframe(result_df)

        # Download results
        csv = result_df.to_csv(index=False).encode('utf-8')
        st.download_button("Download Prediction Results", data=csv, file_name="fraud_predictions.csv", mime="text/csv")

# Footer
st.markdown("---")
st.caption("Credit Card Fraud Detection | Built with Streamlit & XGBoost")