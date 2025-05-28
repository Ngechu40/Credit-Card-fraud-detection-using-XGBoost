**Credit Card Fraud Detection Dashboard**
     **Built with Streamlit & XGBoost**

      **Project Overview**
This project implements a **fraud detection dashboard** using **Streamlit** to visualize and predict fraudulent transactions from a **credit card dataset**. The model is trained using **XGBoost**, leveraging principal component analysis (PCA) features and a transformed **Hour** feature derived from **Time**.

    🎯 **Key Features**
- 📊 **Data Overview**: Display dataset samples for quick reference.
- 📈 **Fraud Analysis**:
  - Fraud vs. Non-Fraud transaction visualization.
  - Distribution of transaction amounts across fraud classes.
  - Fraud occurrences by hour of the day.
- 🔍 **Model Interpretability**:
  - SHAP feature importance for fraud detection transparency.
- ⚡ **Fraud Prediction**:
  - Users can input transaction details and receive **fraud probability predictions**.

    📂 **Dataset**
The dataset includes:
- **PCA-transformed features**: `V1` to `V28`
- **Original features**: `Amount` and `Hour` (converted from `Time`)
- **Target variable**: `Class` (`1 = Fraud, 0 = Non-Fraud`)

    🛠 **Setup Instructions**
   1️⃣ **Install Dependencies**
Ensure Python and `pip` are installed, then run:
```sh
pip install streamlit pandas numpy shap seaborn matplotlib scikit-learn xgboost joblib
```

   2️⃣ **Run the App**
```sh
streamlit run app.py
```

   3️⃣ **Structure**
```
📂 fraud_detection_project/
├── 📄 app.py  # Streamlit application file
├── 📄 creditcard.csv  # Dataset
├── 📄 xgb_best_model.pkl  # Pretrained XGBoost model
├── 📄 README.md  # Documentation file
```

   🔧 **How the Model Works**
- **Preprocessing**: Standardization applied to `Hour`.
- **Training**: XGBoost classifier trained with oversampling for imbalance mitigation.
- **Feature Engineering**: PCA components and derived transaction hour.
- **Prediction**: Fraud likelihood computed from user-input transaction details.

 📌 **Next Steps**
- 🚀 **Enhance model efficiency** with optimized hyperparameters.
- 📊 **Add real-time fraud detection** for live transactions.
- 🧩 **Incorporate interactive data exploration tools**.

  💡 **Contributors**
- **Catherine Muthoni** (Data Scientist & Machine Learning Student)

  📜 **License**
MIT License - Feel free to use and enhance this project.

---

This README provides **installation steps, project details, and usage instructions**.
