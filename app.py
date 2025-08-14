import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model
model = joblib.load('xgboost_model.joblib')

# Define the top 15 features for the UI
top_features = [
    'enq_L3m',
    'Age_Oldest_TL',
    'pct_PL_enq_L6m_of_ever',
    'num_std_12mts',
    'max_recent_level_of_deliq',
    'time_since_recent_enq',
    'recent_level_of_deliq',
    'Age_Newest_TL',
    'num_times_60p_dpd',
    'num_deliq_6_12mts',
    'time_since_recent_payment',
    'first_prod_enq2_others',
    'pct_tl_open_L6M',
    'Home_TL',
    'Unsecured_TL'
]

# Define all features the model was trained on, in the correct order
all_features = [
    'pct_tl_open_L6M', 'pct_tl_closed_L6M', 'Tot_TL_closed_L12M',
    'pct_tl_closed_L12M', 'Tot_Missed_Pmnt', 'CC_TL', 'Home_TL', 'PL_TL',
    'Secured_TL', 'Unsecured_TL', 'Other_TL', 'Age_Oldest_TL',
    'Age_Newest_TL', 'time_since_recent_payment',
    'max_recent_level_of_deliq', 'num_deliq_6_12mts',
    'num_times_60p_dpd', 'num_std_12mts', 'num_sub', 'num_sub_6mts',
    'num_sub_12mts', 'num_dbt', 'num_dbt_12mts', 'num_lss',
    'recent_level_of_deliq', 'CC_enq_L12m', 'PL_enq_L12m',
    'time_since_recent_enq', 'enq_L3m', 'NETMONTHLYINCOME',
    'Time_With_Curr_Empr', 'CC_Flag', 'PL_Flag',
    'pct_PL_enq_L6m_of_ever', 'pct_CC_enq_L6m_of_ever', 'HL_Flag',
    'GL_Flag', 'EDUCATION', 'MARITALSTATUS_Married',
    'MARITALSTATUS_Single', 'GENDER_F', 'GENDER_M', 'last_prod_enq2_AL',
    'last_prod_enq2_CC', 'last_prod_enq2_ConsumerLoan',
    'last_prod_enq2_HL', 'last_prod_enq2_PL', 'last_prod_enq2_others',
    'first_prod_enq2_AL', 'first_prod_enq2_CC',
    'first_prod_enq2_ConsumerLoan', 'first_prod_enq2_HL',
    'first_prod_enq2_PL', 'first_prod_enq2_others'
]

st.title('Credit Risk Prediction')
st.write('This app predicts the credit risk category with 78 percent accuracy for a loan applicant based on the most important features.')

# Create input fields for the top features
input_data = {}
for feature in top_features:
    if feature in ['first_prod_enq2_others', 'Home_TL', 'Unsecured_TL']:
        input_data[feature] = st.selectbox(f'Select {feature}', [0, 1])
    else:
        input_data[feature] = st.number_input(f'Enter {feature}', value=0.0)

# Create a button to make a prediction
if st.button('Predict'):
    # Create a dataframe with all features, initialized to 0
    prediction_df = pd.DataFrame(0, index=[0], columns=all_features)

    # Populate the dataframe with user input
    for feature in top_features:
        prediction_df[feature] = input_data[feature]

    # Make the prediction
    prediction = model.predict(prediction_df)
    prediction_proba = model.predict_proba(prediction_df)
    
    # Decode the prediction
    risk_category_mapping = {0: 'P1', 1: 'P2', 2: 'P3', 3: 'P4'}
    predicted_category = risk_category_mapping[prediction[0]]

    st.success(f'The predicted credit risk category is: {predicted_category}')

    st.write('Prediction Probabilities:')
    proba_df = pd.DataFrame(prediction_proba, columns=[risk_category_mapping[i] for i in range(4)])
    st.write(proba_df)
