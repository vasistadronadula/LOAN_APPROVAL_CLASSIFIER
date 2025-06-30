import streamlit as st
import pandas as pd
import numpy as np
import pickle

# --- Load the trained model and scaler ---
try:
    with open("loan_model.pkl", "rb") as f:
        model, scaler = pickle.load(f)
    # Removed st.sidebar.success as it creates a sidebar element
except FileNotFoundError:
    st.error("Error: 'loan_model.pkl' not found. Please ensure the 'loan_classifier.py' script has been run to train and save the model.")
    st.stop() # Stop the app if model is not found

# Set page config for wide layout and potentially dark theme (if Streamlit's default is dark)
st.set_page_config(page_title="Loan Approval Classifier", layout="wide")

# --- Custom CSS for a better look and feel (Tailwind-like styling) ---
st.markdown("""
<style>
    /* General body and main content styling */
    body {
        color: white; /* Make all text white by default */
        background-color: #1a1a1a; /* Dark background */
    }
    .main {
        background-color: #2c2c2c; /* Slightly lighter dark for main content area */
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
        color: white; /* Ensure text inside main content is also white */
    }

    /* Streamlit specific adjustments for text color */
    .stMarkdown, .stText, .stLabel {
        color: white !important;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #f0f2f6 !important; /* Lighter white for headings */
        text-align: center;
        font-family: 'Inter', sans-serif;
        font-weight: 700;
    }
    h2 {
        text-align: left; /* Keep sub-headings left-aligned */
    }

    /* Button styling */
    .stButton>button {
        background-color: #4CAF50; /* Green */
        color: white;
        padding: 10px 20px;
        border-radius: 8px;
        border: none;
        cursor: pointer;
        font-size: 16px;
        font-weight: bold;
        transition: background-color 0.3s ease, transform 0.2s ease;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
    }
    .stButton>button:hover {
        background-color: #45a049;
        transform: translateY(-2px);
    }
    .stButton>button:active {
        transform: translateY(0);
    }

    /* Input field styling */
    .stTextInput>div>div>input, .stSelectbox>div>div>select, .stNumberInput>div>div>input {
        border-radius: 8px;
        border: 1px solid #555; /* Darker border for dark theme */
        padding: 8px;
        box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.2);
        background-color: #3a3a3a; /* Darker background for inputs */
        color: white; /* White text in inputs */
    }
    /* Style for dropdown options */
    .stSelectbox>div>div>div[role="listbox"] {
        background-color: #3a3a3a;
        color: white;
    }
    .stSelectbox>div>div>div[role="listbox"] div {
        color: white;
    }

    /* Alert styling (success/error messages) */
    .stAlert {
        border-radius: 8px;
        padding: 15px;
        margin-top: 20px;
        font-size: 18px;
        font-weight: bold;
    }
    .stAlert.success {
        background-color: #28a745; /* Darker green for success */
        color: white;
        border: 1px solid #218838;
    }
    .stAlert.error {
        background-color: #ff8c00; /* Dark orange for error/denied status */
        color: white;
        border: 1px solid #cc7000;
    }

    /* Reason box styling */
    .reason-box {
        background-color: #343a40; /* Darker background for reasons */
        /* Removed border-left: 5px solid #007bff; as requested */
        padding: 15px;
        margin-top: 20px;
        border-radius: 5px;
        font-size: 15px;
        color: white; /* White text for reasons */
    }
    .reason-box ul {
        list-style-type: none; /* Remove bullet points */
        padding-left: 0;
    }
    .reason-box li {
        margin-bottom: 5px;
    }
</style>
""", unsafe_allow_html=True)

st.title("Loan Approval Classifier")
st.markdown("---")
st.markdown("Enter the applicant's details below to predict loan approval status.")

# --- Input Features ---
st.header("Applicant Information")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])

with col2:
    applicant_income = st.number_input("Applicant Income (₹)", min_value=0, value=4500, step=100)
    coapplicant_income = st.number_input("Coapplicant Income (₹)", min_value=0, value=0, step=100)
    loan_amount = st.number_input("Loan Amount (₹)", min_value=0, value=120, step=10)
    loan_amount_term = st.selectbox("Loan Amount Term (Days)", [12, 36, 60, 120, 180, 240, 300, 360])
    credit_history = st.selectbox("Credit History (1: Good, 0: Bad)", [1, 0])
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# --- Preprocessing Function (aligned with loan_classifier.py) ---
def preprocess_inputs(gender, married, dependents, education, self_employed,
                      applicant_income, coapplicant_income, loan_amount,
                      loan_amount_term, credit_history, property_area):

    # Manual encoding based on LabelEncoder's alphabetical assignment during training
    # Gender: Female: 0, Male: 1
    gender_encoded = 1 if gender == "Male" else 0
    # Married: No: 0, Yes: 1
    married_encoded = 1 if married == "Yes" else 0
    # Education: Graduate: 0, Not Graduate: 1
    education_encoded = 0 if education == "Graduate" else 1
    # Self_Employed: No: 0, Yes: 1
    self_employed_encoded = 1 if self_employed == "Yes" else 0

    # Dependents mapping (3+ was replaced with 3 in training)
    if dependents == "3+":
        dependents_encoded = 3
    else:
        dependents_encoded = int(dependents)

    # Property Area mapping: Rural: 0, Semiurban: 1, Urban: 2
    property_area_encoded = 0 # Default to Rural
    if property_area == "Semiurban":
        property_area_encoded = 1
    elif property_area == "Urban":
        property_area_encoded = 2

    # Create a dictionary with feature names matching the training data's columns
    # The order of columns in the DataFrame MUST match the X.columns from the training script:
    # ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
    #  'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',
    #  'Credit_History', 'Property_Area']
    features_dict = {
        'Gender': gender_encoded,
        'Married': married_encoded,
        'Dependents': dependents_encoded,
        'Education': education_encoded,
        'Self_Employed': self_employed_encoded,
        'ApplicantIncome': applicant_income,
        'CoapplicantIncome': coapplicant_income,
        'LoanAmount': loan_amount,
        'Loan_Amount_Term': loan_amount_term,
        'Credit_History': credit_history,
        'Property_Area': property_area_encoded
    }

    # Convert to DataFrame with the correct column order
    feature_order = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
                     'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',
                     'Credit_History', 'Property_Area']
    
    input_df = pd.DataFrame([features_dict], columns=feature_order)
    
    return features_dict, input_df # Return both dictionary (for reasoning) and DataFrame (for model)

# --- Function to generate reasons for approval/denial ---
def get_loan_reason(input_data_dict, prediction_status):
    reasons = []
    
    # Common financial indicators
    total_income = input_data_dict['ApplicantIncome'] + input_data_dict['CoapplicantIncome']
    loan_to_income_ratio = input_data_dict['LoanAmount'] / (total_income + 1e-6) # Add small epsilon to avoid division by zero

    if prediction_status == "Approved":
        reasons.append("The loan is likely approved based on the following positive indicators:")
        if input_data_dict['Credit_History'] == 1:
            reasons.append("- **Excellent Credit History:** A strong credit history (evidenced by a score of 1) is a primary factor for approval.")
        if total_income >= 5000: # Example threshold for good income
            reasons.append(f"- **Sufficient Combined Income:** The total household income (₹{total_income:,.0f}) is strong and suggests good repayment capacity.")
        elif input_data_dict['ApplicantIncome'] >= 3000:
             reasons.append(f"- **Good Applicant Income:** The applicant's individual income (₹{input_data_dict['ApplicantIncome']:,.0f}) is supportive.")
        if loan_to_income_ratio < 0.4: # Example threshold for manageable loan amount
            reasons.append(f"- **Manageable Loan Amount:** The requested loan amount (₹{input_data_dict['LoanAmount']:,.0f}) is reasonable relative to the total income ({loan_to_income_ratio:.1%} ratio).")
        if input_data_dict['Property_Area'] == 1: # Semiurban
            reasons.append("- **Favorable Property Location:** Properties in semiurban areas often have better approval rates due to balanced development.")
        elif input_data_dict['Property_Area'] == 2: # Urban
            reasons.append("- **Urban Property Location:** Properties in urban areas are generally well-regarded and offer good collateral.")
        if input_data_dict['Married'] == 1:
            reasons.append("- **Married Status:** Married applicants often demonstrate higher financial stability.")
        if input_data_dict['Education'] == 0: # Graduate
            reasons.append("- **Graduate Education:** Higher education can be a positive indicator of earning potential and stability.")
        if input_data_dict['Self_Employed'] == 0: # Not Self-Employed
            reasons.append("- **Salaried Employment:** Being not self-employed can be seen as more stable and predictable income.")

    else: # Denied
        reasons.append("The loan is likely denied due to the following factors:")
        if input_data_dict['Credit_History'] == 0:
            reasons.append("- **Poor Credit History:** A history of missed payments or defaults (credit score of 0) is a major reason for denial.")
        if total_income < 3000: # Example threshold for low income
            reasons.append(f"- **Low Combined Income:** The total household income (₹{total_income:,.0f}) might be insufficient to comfortably repay the loan.")
        if loan_to_income_ratio >= 0.6: # Example threshold for high loan amount
            reasons.append(f"- **High Loan Amount:** The requested loan amount (₹{input_data_dict['LoanAmount']:,.0f}) is disproportionately high relative to the total income ({loan_to_income_ratio:.1%} ratio).")
        if input_data_dict['Dependents'] >= 2: # High dependents
            reasons.append(f"- **High Number of Dependents:** A larger number of dependents ({input_data_dict['Dependents']}) can strain financial capacity and increase perceived risk.")
        if input_data_dict['Self_Employed'] == 1:
            reasons.append("- **Self-Employment Status:** Self-employment can sometimes be perceived as higher risk without consistent and verifiable income proof.")
        if input_data_dict['Property_Area'] == 0: # Rural
            reasons.append("- **Rural Property Location:** Loans for properties in rural areas may have stricter criteria due to lower liquidity or valuation concerns.")
        if input_data_dict['Loan_Amount_Term'] > 300 and input_data_dict['LoanAmount'] > 150: # Long term for large loan
            reasons.append(f"- **Long Loan Term for Large Amount:** A very long loan term ({input_data_dict['Loan_Amount_Term']} days) for a significant amount might indicate higher repayment risk.")
        if input_data_dict['Married'] == 0:
            reasons.append("- **Unmarried Status:** While not always a sole deciding factor, some models might implicitly favor married applicants for perceived stability.")
        if input_data_dict['Education'] == 1: # Not Graduate
            reasons.append("- **Not Graduate Education:** This might be a minor contributing factor, potentially influencing perceived earning potential.")
            
    return reasons

# --- Prediction Button ---
if st.button("Predict Loan Approval"):
    # Preprocess the inputs
    raw_input_features, processed_input_df = preprocess_inputs(
        gender, married, dependents, education, self_employed,
        applicant_income, coapplicant_income, loan_amount,
        loan_amount_term, credit_history, property_area
    )

    # Scale the input features using the loaded scaler
    scaled_features = scaler.transform(processed_input_df)

    # Make a prediction using the loaded model
    prediction_numeric = model.predict(scaled_features)[0]
    prediction_status = "Approved" if prediction_numeric == 1 else "Rejected"

    # Display the result
    st.markdown("---")
    if prediction_status == "Approved":
        st.markdown(f"<div class='stAlert success'>✅ Loan Status: **{prediction_status}**</div>", unsafe_allow_html=True)
        st.balloons()
    else:
        st.markdown(f"<div class='stAlert error'>❌ Loan Status: **{prediction_status}**</div>", unsafe_allow_html=True)

    # Display reasons
    st.markdown("<br>", unsafe_allow_html=True) # Add some space
    reasons_list = get_loan_reason(raw_input_features, prediction_status)
    st.markdown("<div class='reason-box'>", unsafe_allow_html=True)
    for reason in reasons_list:
        st.markdown(reason, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")
st.markdown("This application uses a pre-trained machine learning model. The reasons provided are based on general financial indicators and common model behaviors, not a direct explanation of the model's internal logic for each specific prediction.")
