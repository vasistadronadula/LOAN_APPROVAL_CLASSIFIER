import streamlit as st
import pandas as pd
import numpy as np
import pickle
import google.generativeai as genai
import re
import json
import time # Keep time for potential future simulations or debugging

# --- HARDCODED GEMINI API KEY ---
# WARNING: For production, use Streamlit Secrets or environment variables for security.
GEMINI_API_KEY = "AIzaSyDjeP9-Uho_ZKJp4Aaiu3C4grBzW5c9R8k"

# --- Configure Gemini API ---
@st.cache_resource
def configure_gemini():
    if GEMINI_API_KEY:
        try:
            genai.configure(api_key=GEMINI_API_KEY)
            return True
        except Exception as e:
            st.error(f"Error configuring Gemini API: {e}. Please check your API key.")
            return False
    else:
        st.error("Gemini API Key is not set. Please provide it in the code or via Streamlit secrets.")
        return False

# --- Load the trained model and scaler ---
try:
    with open("loan_model.pkl", "rb") as f:
        model, scaler = pickle.load(f)
except FileNotFoundError:
    st.error("Error: 'loan_model.pkl' not found. Please ensure the 'loan_classifier.py' script has been run to train and save the model.")
    st.stop()

# Set page config
st.set_page_config(page_title="AI-Powered Loan Approval ", layout="wide")

# --- Custom CSS ---
st.markdown("""
<style>
    body {
        color: white;
        background-color: #1a1a1a;
    }
    .main {
        background-color: #2c2c2c;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
        color: white;
    }
    .stMarkdown, .stText, .stLabel {
        color: white !important;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #f0f2f6 !important;
        text-align: center;
        font-family: 'Inter', sans-serif;
        font-weight: 700;
    }
    h2 {
        text-align: left;
    }
    .stButton>button {
        background-color: #4CAF50;
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
    .stTextInput>div>div>input, .stSelectbox>div>div>select, .stNumberInput>div>div>input {
        border-radius: 8px;
        border: 1px solid #555;
        padding: 8px;
        box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.2);
        background-color: #3a3a3a;
        color: white;
    }
    .stAlert {
        border-radius: 8px;
        padding: 15px;
        margin-top: 20px;
        font-size: 18px;
        font-weight: bold;
    }
    .stAlert.success {
        background-color: #28a745;
        color: white;
        border: 1px solid #218838;
    }
    .stAlert.error {
        background-color: #ff8c00;
        color: white;
        border: 1px solid #cc7000;
    }
    .ai-explanation {
        background-color: #1e3a8a;
        border-left: 5px solid #3b82f6;
        padding: 20px;
        margin-top: 20px;
        border-radius: 8px;
        color: white;
        font-size: 16px;
        line-height: 1.6;
    }
    .natural-language-box {
        background-color: #0f172a;
        border: 2px solid #3b82f6;
        padding: 20px;
        margin: 20px 0;
        border-radius: 10px;
        color: white;
    }

    /* Custom CSS for Spinner */
    .stSpinner {
        display: flex;
        flex-direction: column; /* Stack spinner and text vertically */
        align-items: center;   /* Center horizontally */
        justify-content: center; /* Center vertically (if space allows) */
        padding: 20px; /* Add some padding around the spinner area */
        min-height: 100px; /* Ensure enough space for centering */
    }

    /* Target the text within the spinner */
    .stSpinner > div > div > div:nth-child(2) {
        text-align: center !important;
        width: 100%; /* Ensure text takes full width to center effectively */
        font-size: 1.2em; /* Make text a bit larger */
        color: #f0f2f6; /* Lighter color for visibility */
        margin-top: 10px; /* Space between spinner and text */
    }

    /* Optional: Change spinner color */
    .stSpinner > div > div > div:first-child svg {
        color: #3b82f6 !important; /* Blue color for the spinner */
        width: 3em; /* Make spinner a bit larger */
        height: 3em;
    }
</style>
""", unsafe_allow_html=True)

# --- Initialize session state ---
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'form_input' # Default to form input
if 'extracted_data' not in st.session_state:
    st.session_state.extracted_data = None
if 'prediction_result' not in st.session_state:
    st.session_state.prediction_result = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'last_ai_explanation' not in st.session_state: # To store explanation for persistence
    st.session_state.last_ai_explanation = None

# --- Natural Language Processing Functions ---
def extract_loan_info_from_text(text):
    """Extract loan application information from natural language text using Gemini AI"""
    if not configure_gemini(): # Check if API is configured
        return None
    
    prompt = f"""
    Extract loan application information from the following text and return it as a JSON object.
    
    Text: "{text}"
    
    Please extract the following information and return as JSON:
    {{
        "gender": "Male" or "Female" (default: "Male"),
        "married": "Yes" or "No" (default: "No"),
        "dependents": "0", "1", "2", or "3+" (default: "0"),
        "education": "Graduate" or "Not Graduate" (default: "Graduate"),
        "self_employed": "Yes" or "No" (default: "No"),
        "applicant_income": numeric value (default: 4500),
        "coapplicant_income": numeric value (default: 0),
        "loan_amount": numeric value (default: 120),
        "loan_amount_term": 12, 36, 60, 120, 180, 240, 300, or 360 (default: 360),
        "credit_history": 1 or 0 (1 for good, 0 for bad, default: 1),
        "property_area": "Urban", "Semiurban", or "Rural" (default: "Urban")
    }}
    
    Rules:
    - If information is not explicitly mentioned, use the default values
    - For income values, extract numbers and assume they are in the same currency
    - For loan term, convert years to days (multiply by 30 if mentioned in months, or 365 if in years)
    - Return only valid JSON, no additional text
    """
    
    try:
        model = genai.GenerativeModel("models/gemini-2.5-pro") 
        response = model.generate_content(prompt)
        
        # Clean the response and extract JSON
        json_text = response.text.strip()
        if json_text.startswith('```json'):
            json_text = json_text[7:-3]
        elif json_text.startswith('```'):
            json_text = json_text[3:-3]
        
        return json.loads(json_text)
    except Exception as e:
        st.error(f"Error extracting information: {str(e)}")
        return None

def generate_ai_explanation_content(input_data, prediction_status):
    """Generate AI-powered explanation using Gemini"""
    if not configure_gemini(): # Check if API is configured
        return "AI explanation unavailable. Gemini API not configured."
    
    # Convert input data to readable format
    readable_data = {
        "Gender": input_data.get('gender', 'Male'),
        "Married": input_data.get('married', 'No'),
        "Dependents": input_data.get('dependents', '0'),
        "Education": input_data.get('education', 'Graduate'),
        "Self Employed": input_data.get('self_employed', 'No'),
        "Applicant Income": f"₹{input_data.get('applicant_income', 0):,}",
        "Co-applicant Income": f"₹{input_data.get('coapplicant_income', 0):,}",
        "Loan Amount": f"₹{input_data.get('loan_amount', 0):,}",
        "Loan Term": f"{input_data.get('loan_amount_term', 360)} days",
        "Credit History": "Good" if input_data.get('credit_history', 1) == 1 else "Poor",
        "Property Area": input_data.get('property_area', 'Urban')
    }
    
    prompt = f"""
    As a financial expert, explain why a loan application was {prediction_status.lower()} based on the following applicant information:
    
    Applicant Details:
    {json.dumps(readable_data, indent=2)}
    
    Loan Decision: {prediction_status}
    
    Please provide:
    1. A clear, conversational explanation of the decision
    2. Key factors that influenced the decision
    3. Specific advice for the applicant
    4. If denied, actionable steps to improve their chances in the future
    
    Write in a friendly, professional tone as if you're speaking directly to the applicant.
    Keep the explanation between 150-300 words.
    """
    
    try:
        model = genai.GenerativeModel("models/gemini-2.5-pro") 
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating AI explanation: {str(e)}"

# --- Preprocessing Function ---
def preprocess_inputs(gender, married, dependents, education, self_employed,
                      applicant_income, coapplicant_income, loan_amount,
                      loan_amount_term, credit_history, property_area):
    
    # Manual encoding based on LabelEncoder's alphabetical assignment during training
    gender_encoded = 1 if gender == "Male" else 0
    married_encoded = 1 if married == "Yes" else 0
    education_encoded = 0 if education == "Graduate" else 1
    self_employed_encoded = 1 if self_employed == "Yes" else 0
    
    if dependents == "3+":
        dependents_encoded = 3
    else:
        dependents_encoded = int(dependents)
    
    property_area_encoded = 0  # Default to Rural
    if property_area == "Semiurban":
        property_area_encoded = 1
    elif property_area == "Urban":
        property_area_encoded = 2
    
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
    
    feature_order = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
                     'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term',
                     'Credit_History', 'Property_Area']
    
    input_df = pd.DataFrame([features_dict], columns=feature_order)
    
    return features_dict, input_df

# --- Main App ---
st.title("🤖 AI-Powered Loan Approval Classifier")
st.markdown("---")

# Ensure Gemini API is configured at the start
if not configure_gemini():
    st.warning("Gemini API is not configured. AI features will not be available.")

# --- Menu Bar ---
col1, col2, col3 = st.columns(3)
with col1:
    if st.button("📝 Form Input", use_container_width=True, key="form_btn"):
        st.session_state.current_page = 'form_input'
        st.session_state.prediction_result = None # Clear result when changing input mode
        st.session_state.extracted_data = None
        st.session_state.chat_history = [] # Clear chat history
        st.session_state.last_ai_explanation = None # Clear explanation as well
with col2:
    if st.button("💬 Natural Language Input", use_container_width=True, key="natural_btn"):
        st.session_state.current_page = 'natural_language_input'
        st.session_state.prediction_result = None
        st.session_state.extracted_data = None
        st.session_state.chat_history = [] # Clear chat history
        st.session_state.last_ai_explanation = None # Clear explanation as well
with col3:
    if st.button("❓ AI Query", use_container_width=True, key="ai_query_btn"):
        st.session_state.current_page = 'ai_query'
        # No clearing of prediction_result here, as AI Query needs it
        st.session_state.extracted_data = None
        # st.session_state.chat_history = [] # Don't clear chat history here if we want it to persist when just switching to AI Query tab
        st.session_state.last_ai_explanation = None # Only clear if it's a fresh navigation and no prior explanation
        

st.markdown("---") # Separator below the menu bar

# --- Render Pages based on current_page ---

if st.session_state.current_page == 'form_input':
    ## Form Input
    st.subheader("📝 Enter Applicant Details Manually")
    st.markdown("Fill out the form below to get a loan approval prediction.")
    
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
    
    if st.button("🔮 Predict Loan Approval", use_container_width=True):
        # Preprocess the inputs
        raw_input_features, processed_input_df = preprocess_inputs(
            gender, married, dependents, education, self_employed,
            applicant_income, coapplicant_income, loan_amount,
            loan_amount_term, credit_history, property_area
        )
        
        # Make prediction
        scaled_features = scaler.transform(processed_input_df)
        prediction_numeric = model.predict(scaled_features)[0]
        prediction_status = "Approved" if prediction_numeric == 1 else "Rejected"
        
        # Store result
        form_data = {
            'gender': gender,
            'married': married,
            'dependents': dependents,
            'education': education,
            'self_employed': self_employed,
            'applicant_income': applicant_income,
            'coapplicant_income': coapplicant_income,
            'loan_amount': loan_amount,
            'loan_amount_term': loan_amount_term,
            'credit_history': credit_history,
            'property_area': property_area
        }
        
        st.session_state.prediction_result = {
            'status': prediction_status,
            'input_data': form_data,
            'raw_features': raw_input_features
        }
        st.session_state.last_ai_explanation = None # Clear previous explanation when new prediction is made
        st.session_state.chat_history = [] # Clear chat history on new prediction
        st.session_state.current_page = 'ai_query' # Automatically navigate to AI Query tab
        st.rerun() # Rerun to navigate to the new page

elif st.session_state.current_page == 'natural_language_input':
    ## Natural Language Input
    st.markdown('<div class="natural-language-box">', unsafe_allow_html=True)
    st.subheader("💬 Describe Your Loan Application Naturally")
    st.markdown("Example: *I'm a 35-year-old married software engineer earning ₹80,000 monthly. My wife earns ₹40,000. We want a ₹2,500,000 loan for 20 years to buy a house in Mumbai. I have a good credit history and no dependents.*")
    
    natural_input = st.text_area(
        "Describe your loan application:",
        placeholder="Tell us about your income, family situation, loan requirements, credit history, etc.",
        height=120,
        key="natural_input_text_area"
    )
    
    if st.button("🧠 Extract Information with AI", use_container_width=True):
        if natural_input.strip():
            with st.spinner("Extracting information using AI..."):
                extracted_data = extract_loan_info_from_text(natural_input)
                if extracted_data:
                    st.session_state.extracted_data = extracted_data
                    st.success("✅ Information extracted successfully! Review below before predicting.")
                else:
                    st.error("❌ Could not extract information. Please try again or use form input.")
        else:
            st.warning("Please enter some text about your loan application.")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Display extracted information
    if st.session_state.extracted_data:
        st.subheader("📋 Extracted Information Review")
        st.info("Please review the extracted details. If anything is incorrect, adjust your natural language input and try again.")
        col1, col2 = st.columns(2)
        
        data = st.session_state.extracted_data
        with col1:
            st.write(f"**Gender:** {data.get('gender', 'Male')}")
            st.write(f"**Married:** {data.get('married', 'No')}")
            st.write(f"**Dependents:** {data.get('dependents', '0')}")
            st.write(f"**Education:** {data.get('education', 'Graduate')}")
            st.write(f"**Self Employed:** {data.get('self_employed', 'No')}")
            st.write(f"**Property Area:** {data.get('property_area', 'Urban')}")
        
        with col2:
            st.write(f"**Applicant Income:** ₹{data.get('applicant_income', 0):,}")
            st.write(f"**Co-applicant Income:** ₹{data.get('coapplicant_income', 0):,}")
            st.write(f"**Loan Amount:** ₹{data.get('loan_amount', 0):,}")
            st.write(f"**Loan Term:** {data.get('loan_amount_term', 360)} days")
            st.write(f"**Credit History:** {'Good' if data.get('credit_history', 1) == 1 else 'Poor'}")
        
        # Use extracted data for prediction
        if st.button("🔮 Predict Loan Approval with Extracted Data", use_container_width=True):
            # Process prediction with extracted data
            raw_input_features, processed_input_df = preprocess_inputs(
                data['gender'], data['married'], data['dependents'], 
                data['education'], data['self_employed'],
                data['applicant_income'], data['coapplicant_income'], 
                data['loan_amount'], data['loan_amount_term'], 
                data['credit_history'], data['property_area']
            )
            
            # Make prediction
            scaled_features = scaler.transform(processed_input_df)
            prediction_numeric = model.predict(scaled_features)[0]
            prediction_status = "Approved" if prediction_numeric == 1 else "Rejected"
            
            # Store result
            st.session_state.prediction_result = {
                'status': prediction_status,
                'input_data': data,
                'raw_features': raw_input_features
            }
            st.session_state.last_ai_explanation = None # Clear previous explanation
            st.session_state.chat_history = [] # Clear chat history on new prediction
            st.session_state.current_page = 'ai_query' # Automatically navigate to AI Query tab
            st.rerun() # Rerun to navigate to the new page

elif st.session_state.current_page == 'ai_query':
    ## AI Query & Previous Results
    st.subheader("❓ AI Insights and Live Chat")
    st.markdown("Here you can get an **AI explanation** for the *last loan prediction* made and **chat live** with the AI about your application or general loan queries.")

    if st.session_state.prediction_result:
        result = st.session_state.prediction_result
        st.markdown("---")
        st.markdown("#### Last Loan Decision:")
        if result['status'] == "Approved":
            st.markdown(f"<div class='stAlert success'>✅ Loan Status: **{result['status']}**</div>", unsafe_allow_html=True)
            # st.balloons() # Removed balloons
        else:
            st.markdown(f"<div class='stAlert error'>❌ Loan Status: **{result['status']}**</div>", unsafe_allow_html=True)

        st.markdown("### 🤖 AI Explanation")
        
        # Use st.button and st.spinner for the AI explanation generation
        if st.session_state.last_ai_explanation:
            st.markdown(f'<div class="ai-explanation">{st.session_state.last_ai_explanation}</div>', unsafe_allow_html=True)
        elif st.button("Generate AI Explanation", key="generate_explanation_btn", use_container_width=True):
            # The spinner will cover the entire duration of the AI call
            with st.spinner("Generating detailed AI explanation... Please wait, this might take a moment."):
                ai_explanation = generate_ai_explanation_content(
                    result['input_data'], 
                    result['status']
                )
                st.session_state.last_ai_explanation = ai_explanation # Store for persistence
            st.rerun() # Rerun to display the generated explanation now that it's in session_state
    else:
        st.info("No loan prediction has been made yet. Please use 'Form Input' or 'Natural Language Input' first to get a prediction.")
    
    # Chat with AI about the result or general questions
    if configure_gemini():
        st.markdown("---")
        st.markdown("### 💬 Chat with AI Assistant")
        
        # Use a form for the chat input to handle submission more cleanly
        with st.form("chat_form", clear_on_submit=True):
            user_question = st.text_input("Ask a question about the last prediction, or general loan advice:", key="chat_input_text_area")
            submit_button = st.form_submit_button("Send")

            if submit_button and user_question:
                # Construct context for the AI
                chat_context = ""
                if st.session_state.prediction_result:
                    chat_context += f"""
                    The last loan application had these details:
                    {json.dumps(st.session_state.prediction_result['input_data'], indent=2)}
                    The decision was: {st.session_state.prediction_result['status']}.
                    """
                
                prompt_for_chat = f"""
                You are a helpful AI assistant specializing in loan applications and financial advice.
                {chat_context}
                
                User Question: {user_question}
                
                Please provide a helpful, accurate, and concise response. Avoid making predictions about new scenarios.
                """
                
                try:
                    with st.spinner("AI is thinking..."): # Spinner for chat response
                        model = genai.GenerativeModel("models/gemini-2.5-pro") 
                        chat_response = model.generate_content(prompt_for_chat)
                        
                        # Add to chat history
                        st.session_state.chat_history.append({
                            'question': user_question,
                            'answer': chat_response.text
                        })
                    
                except Exception as e:
                    st.error(f"Error generating chat response: {str(e)}")
                    
                st.rerun() # This will refresh the page to show the new chat message

        # Display chat history
        if st.session_state.chat_history:
            st.markdown("#### Chat History:")
            # Display chat messages in a more conversational flow
            for chat in reversed(st.session_state.chat_history): # Display latest message first
                st.markdown(f"**You:** {chat['question']}")
                st.markdown(f"**AI:** {chat['answer']}")
                st.markdown("---")

st.markdown("---")
st.markdown("This application uses a pre-trained machine learning model enhanced with Gemini AI for natural language processing and explanations.")
