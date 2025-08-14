# Credit Risk Modeling using Machine Learning

This project is a comprehensive, end-to-end demonstration of a machine learning solution for predicting credit risk. The goal is to assess the likelihood of a loan applicant defaulting, enabling a financial institution to make data-driven decisions. The final output is an interactive web application built with Streamlit that allows for real-time risk assessment based on key applicant data.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Methodology](#methodology)
- [Key Predictive Features](#key-predictive-features)
- [Getting Started](#getting-started)
- [Live Demonstration](#live-demonstration)
- [Future Improvements](#future-improvements)

---

## Project Overview

In the financial sector, accurately assessing the risk of lending is paramount. This project tackles this challenge by building a robust classification model that categorizes loan applicants into four priority levels (P1, P2, P3, P4) based on their creditworthiness. The project covers the entire machine learning lifecycle:

- **Data Cleaning & Preprocessing:** Handling missing values and preparing raw data for analysis.
- **Exploratory Data Analysis (EDA):** Using statistical tests (Chi-square, VIF, ANOVA) to understand feature relevance and multicollinearity.
- **Feature Engineering & Selection:** Creating new features and selecting the most impactful variables for the model.
- **Model Training & Comparison:** Evaluating multiple algorithms (Random Forest, Decision Tree, XGBoost) to find the best performer.
- **Hyperparameter Tuning:** Optimizing the chosen model (XGBoost) to maximize its predictive accuracy.
- **Deployment:** Serving the final model through an interactive Streamlit web application.

## Tech Stack

- **Programming Language:** Python 3
- **Data Manipulation & Analysis:** Pandas, NumPy
- **Data Visualization:** Matplotlib
- **Machine Learning:** Scikit-learn, XGBoost
- **Statistical Analysis:** SciPy, Statsmodels
- **Web Application:** Streamlit
- **Model Persistence:** Joblib

## Project Structure

```
├── Credit_risk_notebook.ipynb      # Main Jupyter Notebook with all analysis and model building
├── app.py                          # The Streamlit web application script
├── requirements.txt                # Pip requirements for the Streamlit app
├── notebook-requirements.txt       # Pip requirements for the Jupyter notebook
├── xgboost_model.joblib            # The final, trained XGBoost model
├── .gitignore                      # Specifies files to be ignored by Git
├── dataset/
│   ├── case_study1.xlsx          # Raw data file 1
│   └── case_study2.xlsx          # Raw data file 2
└── README.md                       # Project documentation
```

## Methodology

The project followed a structured, multi-stage approach to ensure a robust and well-documented outcome.

1. **Data Cleaning:** The two raw datasets were merged. Missing values, represented by `-99999`, were systematically handled. Columns with a high percentage of missing data were dropped, and remaining invalid entries were filtered out.
2. **Feature Selection:** To build an efficient and interpretable model, a rigorous feature selection process was employed:

   - **Categorical Features:** A Chi-square test was used to assess the statistical significance of categorical variables against the target variable (`Approved_Flag`).
   - **Numerical Features:** The Variance Inflation Factor (VIF) was used to identify and remove features with high multicollinearity. Subsequently, an ANOVA F-test was performed to select the numerical features that have a statistically significant relationship with the target.
3. **Model Evaluation:** Three different classification models were trained and evaluated based on their accuracy, precision, recall, and F1-score.

   - Random Forest: 76% Accuracy
   - Decision Tree: 71% Accuracy
   - **XGBoost: 78% Accuracy**

   XGBoost was selected as the champion model due to its superior performance.
4. **Hyperparameter Tuning:** The selected XGBoost model was further optimized using `GridSearchCV` to find the best combination of hyperparameters, resulting in a final test accuracy of **78.12%**.

## Key Predictive Features

The final model revealed several key drivers in predicting credit risk. The top 5 most influential features are:

1. **`enq_L3m` (Enquiries in last 3 months):** A high number of recent credit enquiries often signals financial distress, making it a strong predictor of risk.
2. **`Age_Oldest_TL` (Age of oldest trade line):** A long credit history is generally a sign of stability and lower risk.
3. **`pct_PL_enq_L6m_of_ever` (Percentage of recent personal loan enquiries):** A sudden spike in personal loan applications can be a red flag.
4. **`num_std_12mts` (Number of standard payments in last 12 months):** This directly measures the applicant's recent payment behavior and reliability.
5. **`max_recent_level_of_deliq` (Max recent delinquency):** This shows how severely the applicant has missed recent payments, indicating higher risk.

## Getting Started

To run this project locally, please follow these steps:

1. **Prerequisites:**

   - Python 3.8 or higher
   - Git
2. **Clone the repository:**

   ```bash
   git clone https://github.com/780Noman/Credit_Risk_assesment.git
   cd Credit_Risk_assesment
   ```
3. **Set up a virtual environment (recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```
4. **Install dependencies for the Streamlit app:**

   ```bash
   pip install -r requirements.txt
   ```
5. **Run the Streamlit Application:**

   ```bash
   streamlit run app.py
   ```
6. **(Optional) To run the analysis notebook:**
   If you wish to explore the data analysis and model building process, install the full set of dependencies and run the Jupyter Notebook.

   ```bash
   pip install -r notebook-requirements.txt
   jupyter notebook Credit_risk_notebook.ipynb
   ```

## Live Demonstration

Below is a screenshot of the interactive Streamlit application. The interface is designed to be simple and intuitive, allowing users to input key financial metrics and receive an instant credit risk assessment.

*(Placeholder: You can add a screenshot of your running Streamlit app here. Drag and drop the image into the GitHub text editor.)*


## Future Improvements

This project provides a solid foundation for a credit risk model. Future enhancements could include:

- **Advanced Feature Engineering:** Exploring more complex interactions between features.
- **Alternative Modeling Techniques:** Experimenting with other algorithms like LightGBM or CatBoost, or using deep learning models.
- **Scalable Deployment:** Containerizing the application with Docker and deploying it as a scalable REST API using a framework like FastAPI.
- **Explainable AI (XAI):** Integrating tools like SHAP or LIME to provide clear, human-understandable explanations for each individual prediction.
