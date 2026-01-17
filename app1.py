import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
import json
import time
import os
from datetime import datetime
from fpdf import FPDF
import plotly.graph_objects as go
from typing import Dict, Any, Optional, Tuple
import logging

# =========================================================
# LOGGING CONFIGURATION
# =========================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="Loan Approval AI System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =========================================================
# CONFIG & CONSTANTS
# =========================================================
try:
    OPENROUTER_API_KEY = st.secrets.get("OPENROUTER_API_KEY", "")
    OPENROUTER_MODEL = st.secrets.get("OPENROUTER_MODEL", "openai/gpt-4o-mini")
    APP_URL = st.secrets.get("APP_URL", "http://localhost:8501")
    APP_NAME = st.secrets.get("APP_NAME", "Loan Approval AI System")
except Exception as e:
    logger.warning(f"Error loading secrets: {e}")
    OPENROUTER_API_KEY = ""
    OPENROUTER_MODEL = "openai/gpt-4o-mini"
    APP_URL = "http://localhost:8501"
    APP_NAME = "Loan Approval AI System"

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Build headers safely
OPENROUTER_HEADERS = {
    "Content-Type": "application/json",
    "HTTP-Referer": APP_URL,
    "X-Title": APP_NAME,
}
if OPENROUTER_API_KEY:
    OPENROUTER_HEADERS["Authorization"] = f"Bearer {OPENROUTER_API_KEY}"


def has_valid_openrouter_key() -> bool:
    """Check if OpenRouter API key is valid."""
    return bool(OPENROUTER_API_KEY and OPENROUTER_API_KEY.startswith("sk-or-v1-"))


# =========================================================
# MODEL LOADING
# =========================================================
@st.cache_resource
def load_model() -> Tuple[Any, Any]:
    """Load the trained ML model and scaler with error handling."""
    try:
        with open("loan_model.pkl", "rb") as f:
            model, scaler = pickle.load(f)
        logger.info("Model loaded successfully")
        return model, scaler
    except FileNotFoundError:
        logger.error("Model file not found")
        st.error("❌ Model file not found. Please ensure 'loan_model.pkl' exists in this directory.")
        st.stop()
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        st.error(f"❌ Error loading model: {e}")
        st.stop()


try:
    model, scaler = load_model()
except Exception as e:
    logger.critical(f"Failed to initialize model: {e}")
    st.error("Critical error: Unable to load model. Please check the logs.")
    st.stop()

# =========================================================
# PREMIUM BLACK THEME CSS
# =========================================================
st.markdown(
    """
<style>
    /* Global Styles */
    .stApp {
        background: linear-gradient(135deg, #000000 0%, #0a0a0a 25%, #1a1a2e 50%, #0f0f23 75%, #000000 100%);
        color: #e5e7eb;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Main Header */
    .main-header {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #0f172a 100%);
        padding: 2.5rem;
        border-radius: 0 0 32px 32px;
        margin-bottom: 2.5rem;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.9), 0 0 100px rgba(59, 130, 246, 0.1);
        border-bottom: 2px solid rgba(59, 130, 246, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .main-header::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: linear-gradient(45deg, transparent 0%, rgba(59, 130, 246, 0.05) 50%, transparent 100%);
        animation: shimmer 3s infinite;
    }
    
    @keyframes shimmer {
        0%, 100% { transform: translateX(-100%); }
        50% { transform: translateX(100%); }
    }
    
    .metric-pill {
        background: rgba(15, 23, 42, 0.9);
        border-radius: 999px;
        padding: 0.5rem 1.2rem;
        border: 1px solid rgba(59, 130, 246, 0.5);
        font-size: 0.85rem;
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.2);
        backdrop-filter: blur(10px);
    }
    
    /* Card Styles */
    .card {
        background: linear-gradient(135deg, #0f172a 0%, #0a0a0a 100%);
        border-radius: 24px;
        padding: 2rem;
        border: 1px solid rgba(59, 130, 246, 0.3);
        box-shadow: 0 20px 50px rgba(0, 0, 0, 0.8), 
                    0 0 0 1px rgba(59, 130, 246, 0.1) inset;
        margin-bottom: 1.5rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, rgba(59, 130, 246, 0.5), transparent);
    }
    
    .card:hover {
        transform: translateY(-4px);
        box-shadow: 0 30px 70px rgba(0, 0, 0, 0.9), 
                    0 0 0 1px rgba(59, 130, 246, 0.4) inset;
        border-color: rgba(59, 130, 246, 0.6);
    }

    /* Button Styles */
    .stButton > button {
        background: linear-gradient(135deg, #2563eb 0%, #4f46e5 50%, #7c3aed 100%);
        color: #ffffff;
        border: none;
        border-radius: 999px;
        padding: 0.85rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 10px 30px rgba(37, 99, 235, 0.5), 
                    0 0 0 1px rgba(59, 130, 246, 0.3) inset;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) scale(1.02);
        box-shadow: 0 20px 50px rgba(59, 130, 246, 0.8), 
                    0 0 0 1px rgba(59, 130, 246, 0.6) inset;
        background: linear-gradient(135deg, #1d4ed8 0%, #4338ca 50%, #6d28d9 100%);
    }
    
    .stButton > button:active {
        transform: translateY(0) scale(0.98);
    }

    /* Input Styles */
    .stTextInput > div > div > input,
    .stSelectbox > div > div > select,
    .stNumberInput > div > div > input,
    textarea {
        border-radius: 16px !important;
        border: 1px solid rgba(59, 130, 246, 0.3) !important;
        padding: 0.75rem 1.25rem !important;
        font-size: 0.95rem !important;
        background: rgba(15, 23, 42, 0.95) !important;
        color: #e5e7eb !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.5) inset !important;
    }
    
    .stTextInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus,
    .stNumberInput > div > div > input:focus,
    textarea:focus {
        border-color: rgba(59, 130, 246, 0.8) !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.2), 
                    0 4px 12px rgba(0, 0, 0, 0.5) inset !important;
        outline: none !important;
        background: rgba(15, 23, 42, 1) !important;
    }

    /* Status Cards */
    .status-approved {
        background: linear-gradient(135deg, #059669 0%, #047857 50%, #065f46 100%);
        color: #ffffff;
        padding: 3rem;
        border-radius: 28px;
        text-align: center;
        animation: pulseGreen 2s infinite;
        border: 2px solid rgba(16, 185, 129, 0.5);
        box-shadow: 0 25px 70px rgba(16, 185, 129, 0.6), 
                    0 0 0 1px rgba(16, 185, 129, 0.3) inset;
        position: relative;
        overflow: hidden;
    }
    
    .status-approved::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255, 255, 255, 0.1) 0%, transparent 70%);
        animation: rotate 8s linear infinite;
    }
    
    .status-rejected {
        background: linear-gradient(135deg, #dc2626 0%, #b91c1c 50%, #991b1b 100%);
        color: #ffffff;
        padding: 3rem;
        border-radius: 28px;
        text-align: center;
        border: 2px solid rgba(239, 68, 68, 0.5);
        box-shadow: 0 25px 70px rgba(239, 68, 68, 0.6), 
                    0 0 0 1px rgba(239, 68, 68, 0.3) inset;
    }
    
    @keyframes pulseGreen {
        0%, 100% { 
            box-shadow: 0 25px 70px rgba(16, 185, 129, 0.6), 
                        0 0 0 1px rgba(16, 185, 129, 0.3) inset; 
        }
        50% { 
            box-shadow: 0 25px 70px rgba(16, 185, 129, 0.9), 
                        0 0 0 1px rgba(16, 185, 129, 0.5) inset,
                        0 0 60px rgba(16, 185, 129, 0.4); 
        }
    }
    
    @keyframes rotate {
        from { transform: rotate(0deg); }
        to { transform: rotate(360deg); }
    }

    /* Tab Styles */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
        background: transparent;
        border-bottom: 2px solid rgba(59, 130, 246, 0.2);
        padding-bottom: 0;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 1rem 2rem;
        border-radius: 16px 16px 0 0;
        background: rgba(15, 23, 42, 0.5);
        color: #9ca3af;
        border: 1px solid transparent;
        transition: all 0.3s ease;
        font-weight: 500;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(15, 23, 42, 0.8);
        color: #e5e7eb;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #1e40af 0%, #4f46e5 100%) !important;
        color: #ffffff !important;
        border-color: rgba(59, 130, 246, 0.8) !important;
        box-shadow: 0 8px 24px rgba(59, 130, 246, 0.4);
    }

    /* Chat Bubbles */
    .chat-user {
        background: linear-gradient(135deg, #2563eb 0%, #4f46e5 100%);
        color: white;
        border-radius: 20px 20px 4px 20px;
        padding: 1rem 1.5rem;
        margin: 0.75rem 0;
        max-width: 85%;
        margin-left: auto;
        box-shadow: 0 12px 32px rgba(37, 99, 235, 0.5);
        animation: slideInRight 0.3s ease;
    }
    
    .chat-ai {
        background: rgba(15, 23, 42, 0.95);
        color: #e5e7eb;
        border-radius: 20px 20px 20px 4px;
        padding: 1rem 1.5rem;
        margin: 0.75rem 0;
        max-width: 85%;
        border: 1px solid rgba(59, 130, 246, 0.3);
        box-shadow: 0 12px 32px rgba(0, 0, 0, 0.6);
        animation: slideInLeft 0.3s ease;
    }
    
    @keyframes slideInRight {
        from { opacity: 0; transform: translateX(20px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    @keyframes slideInLeft {
        from { opacity: 0; transform: translateX(-20px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: #0a0a0a;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #2563eb 0%, #4f46e5 100%);
        border-radius: 5px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #1d4ed8 0%, #4338ca 100%);
    }
    
    /* Loading Animation */
    .stSpinner > div {
        border-color: rgba(59, 130, 246, 0.3) !important;
        border-top-color: #3b82f6 !important;
    }
    
    /* Fade in animation */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Success/Error Messages */
    .stSuccess {
        background: rgba(16, 185, 129, 0.1) !important;
        border: 1px solid rgba(16, 185, 129, 0.3) !important;
        border-radius: 16px !important;
        color: #10b981 !important;
    }
    
    .stError {
        background: rgba(239, 68, 68, 0.1) !important;
        border: 1px solid rgba(239, 68, 68, 0.3) !important;
        border-radius: 16px !important;
        color: #ef4444 !important;
    }
    
    .stWarning {
        background: rgba(245, 158, 11, 0.1) !important;
        border: 1px solid rgba(245, 158, 11, 0.3) !important;
        border-radius: 16px !important;
        color: #f59e0b !important;
    }
    
    .stInfo {
        background: rgba(59, 130, 246, 0.1) !important;
        border: 1px solid rgba(59, 130, 246, 0.3) !important;
        border-radius: 16px !important;
        color: #3b82f6 !important;
    }
    
    /* Results Section */
    .results-section {
        border-top: 2px solid rgba(59, 130, 246, 0.3);
        padding-top: 2rem;
        margin-top: 3rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

# =========================================================
# SESSION STATE INITIALIZATION
# =========================================================
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None
if "form_data" not in st.session_state:
    st.session_state.form_data = {}
if "extracted_info" not in st.session_state:
    st.session_state.extracted_info = None
if "pdf_bytes" not in st.session_state:
    st.session_state.pdf_bytes = None
if "ai_recommendations" not in st.session_state:
    st.session_state.ai_recommendations = None
if "active_tab" not in st.session_state:
    st.session_state.active_tab = 0
if "structured_result" not in st.session_state:
    st.session_state.structured_result = None
if "ai_result" not in st.session_state:
    st.session_state.ai_result = None
if "ai_extracted_data" not in st.session_state:
    st.session_state.ai_extracted_data = None

# =========================================================
# CORE ML FUNCTIONS
# =========================================================
def preprocess_inputs(
    gender: str,
    married: str,
    dependents: str,
    education: str,
    self_employed: str,
    applicant_income: int,
    coapplicant_income: int,
    loan_amount: int,
    loan_amount_term: int,
    credit_history: int,
    property_area: str,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Preprocess input data for ML model with validation."""
    try:
        encodings = {
            "gender": 1 if gender == "Male" else 0,
            "married": 1 if married == "Yes" else 0,
            "education": 0 if education == "Graduate" else 1,
            "self_employed": 1 if self_employed == "Yes" else 0,
            "dependents": 3 if dependents == "3+" else int(dependents),
            "property_area": {"Rural": 0, "Semiurban": 1, "Urban": 2}[property_area],
        }

        features = {
            "Gender": encodings["gender"],
            "Married": encodings["married"],
            "Dependents": encodings["dependents"],
            "Education": encodings["education"],
            "Self_Employed": encodings["self_employed"],
            "ApplicantIncome": int(applicant_income),
            "CoapplicantIncome": int(coapplicant_income),
            "LoanAmount": int(loan_amount),
            "Loan_Amount_Term": int(loan_amount_term),
            "Credit_History": int(credit_history),
            "Property_Area": encodings["property_area"],
        }

        feature_order = [
            "Gender", "Married", "Dependents", "Education", "Self_Employed",
            "ApplicantIncome", "CoapplicantIncome", "LoanAmount",
            "Loan_Amount_Term", "Credit_History", "Property_Area",
        ]
        
        input_df = pd.DataFrame([features], columns=feature_order)
        logger.info("Input preprocessing successful")
        return features, input_df
    
    except Exception as e:
        logger.error(f"Error in preprocessing: {e}")
        raise ValueError(f"Error preprocessing inputs: {e}")


def make_prediction(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Make loan approval prediction with comprehensive error handling."""
    try:
        features, df = preprocess_inputs(**input_data)
        scaled = scaler.transform(df)
        prediction = model.predict(scaled)[0]

        try:
            probability = float(model.predict_proba(scaled)[0].max())
        except Exception as e:
            logger.warning(f"Could not get probability: {e}")
            probability = 0.75

        result = {
            "status": "Approved" if prediction == 1 else "Rejected",
            "confidence": probability,
            "input_data": input_data,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        
        logger.info(f"Prediction made: {result['status']} with confidence {probability:.2%}")
        return result
    
    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        raise RuntimeError(f"Prediction failed: {e}")


# =========================================================
# OPENROUTER AI FUNCTIONS
# =========================================================
def call_openrouter_system(
    user_prompt: str, 
    system_prompt: str, 
    temperature: float = 0.2, 
    max_tokens: int = 512
) -> str:
    """Call OpenRouter API with comprehensive error handling."""
    if not has_valid_openrouter_key():
        raise RuntimeError("OpenRouter API key is missing or invalid.")

    try:
        payload = {
            "model": OPENROUTER_MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        resp = requests.post(
            OPENROUTER_URL, 
            headers=OPENROUTER_HEADERS, 
            data=json.dumps(payload), 
            timeout=60
        )
        
        if resp.status_code != 200:
            logger.error(f"OpenRouter API error: {resp.status_code} - {resp.text}")
            raise RuntimeError(f"OpenRouter API error: {resp.status_code}")

        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        logger.info("OpenRouter API call successful")
        return content
    
    except requests.exceptions.Timeout:
        logger.error("OpenRouter API timeout")
        raise RuntimeError("Request timed out. Please try again.")
    except requests.exceptions.RequestException as e:
        logger.error(f"OpenRouter request error: {e}")
        raise RuntimeError(f"Network error: {e}")
    except Exception as e:
        logger.error(f"Unexpected error in OpenRouter call: {e}")
        raise RuntimeError(f"AI service error: {e}")


def ai_extract_features_from_text(raw_text: str) -> Dict[str, Any]:
    """Extract loan features from natural language using AI."""
    system_prompt = (
        "You are an assistant that extracts loan application features as strict JSON. "
        "Return ONLY valid JSON, no explanations or markdown. Fields: "
        "gender (Male/Female), married (Yes/No), dependents (0/1/2/3+), "
        "education (Graduate/Not Graduate), self_employed (Yes/No), "
        "applicant_income (int, monthly INR), coapplicant_income (int), "
        "loan_amount (int total INR), loan_amount_term (int months), "
        "credit_history (0 or 1), property_area (Urban/Semiurban/Rural)."
    )

    user_prompt = (
        f"Description:\n{raw_text}\n\n"
        "Extract and output JSON in this exact schema:\n"
        "{\n"
        '  "gender": "Male",\n'
        '  "married": "Yes",\n'
        '  "dependents": "0",\n'
        '  "education": "Graduate",\n'
        '  "self_employed": "No",\n'
        '  "applicant_income": 85000,\n'
        '  "coapplicant_income": 0,\n'
        '  "loan_amount": 2500000,\n'
        '  "loan_amount_term": 240,\n'
        '  "credit_history": 1,\n'
        '  "property_area": "Urban"\n'
        "}"
    )

    try:
        content = call_openrouter_system(user_prompt, system_prompt, temperature=0.1, max_tokens=400)

        # Try to parse JSON
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            # Extract JSON from markdown or text
            start = content.find("{")
            end = content.rfind("}") + 1
            if start == -1 or end == 0:
                raise ValueError("No JSON found in response")
            data = json.loads(content[start:end])

        # Default values
        defaults = {
            "gender": "Male",
            "married": "No",
            "dependents": "0",
            "education": "Graduate",
            "self_employed": "No",
            "applicant_income": 50000,
            "coapplicant_income": 0,
            "loan_amount": 500000,
            "loan_amount_term": 240,
            "credit_history": 1,
            "property_area": "Urban",
        }

        # Clean and validate data
        cleaned = {}
        for k, v in defaults.items():
            cleaned[k] = data.get(k, v)

        # Type conversion
        cleaned["dependents"] = str(cleaned["dependents"])
        cleaned["applicant_income"] = int(cleaned["applicant_income"])
        cleaned["coapplicant_income"] = int(cleaned["coapplicant_income"])
        cleaned["loan_amount"] = int(cleaned["loan_amount"])
        cleaned["loan_amount_term"] = int(cleaned["loan_amount_term"])
        cleaned["credit_history"] = int(cleaned["credit_history"])

        logger.info("Feature extraction successful")
        return cleaned
    
    except Exception as e:
        logger.error(f"Error extracting features: {e}")
        raise RuntimeError(f"Failed to extract features from text: {e}")


def ai_recommendations(input_data: Dict[str, Any], result: Dict[str, Any]) -> str:
    """Generate AI-powered recommendations."""
    system_prompt = (
        "You are a senior credit risk officer at a modern fintech company. "
        "Provide clear, actionable recommendations in simple language. "
        "Format your response with bullet points using markdown."
    )

    user_prompt = (
        "Loan Decision Analysis:\n\n"
        f"Application Data:\n{json.dumps(input_data, indent=2)}\n\n"
        f"Decision: {result['status']}\n"
        f"Confidence: {result['confidence']:.1%}\n\n"
        "Please provide:\n"
        "1. Brief explanation of the decision (2-3 sentences)\n"
        "2. 3-5 specific, actionable recommendations to improve approval chances or manage the loan\n\n"
        "Use bullet points for recommendations."
    )

    try:
        recommendations = call_openrouter_system(
            user_prompt, 
            system_prompt, 
            temperature=0.3, 
            max_tokens=600
        )
        logger.info("AI recommendations generated")
        return recommendations
    except Exception as e:
        logger.error(f"Error generating recommendations: {e}")
        raise RuntimeError(f"Failed to generate recommendations: {e}")


# =========================================================
# VISUALIZATION FUNCTIONS
# =========================================================
def approval_gauge(prob: float) -> go.Figure:
    """Create approval confidence gauge chart."""
    try:
        pct = float(prob) * 100.0
        
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=pct,
                number={"suffix": "%", "font": {"color": "#e5e7eb", "size": 32}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#9ca3af", "tickfont": {"color": "#9ca3af"}},
                    "bar": {"color": "#10b981", "thickness": 0.7},
                    "bgcolor": "#0a0a0a",
                    "borderwidth": 2,
                    "bordercolor": "#374151",
                    "steps": [
                        {"range": [0, 40], "color": "#450a0a"},
                        {"range": [40, 70], "color": "#1f2937"},
                        {"range": [70, 100], "color": "#064e3b"},
                    ],
                    "threshold": {
                        "line": {"color": "#fbbf24", "width": 5},
                        "thickness": 0.8,
                        "value": pct,
                    },
                },
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Approval Confidence", "font": {"color": "#e5e7eb", "size": 20}},
            )
        )
        
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=20, r=20, t=60, b=20),
            height=300,
        )
        
        return fig
    except Exception as e:
        logger.error(f"Error creating gauge: {e}")
        return go.Figure()


def dummy_approval_trend(current_conf: float) -> go.Figure:
    """Create approval trend comparison chart."""
    try:
        base = np.linspace(0.55, 0.78, 6)
        history = list(base) + [current_conf]
        labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "You"]

        fig = go.Figure()
        
        # Add average line
        fig.add_trace(
            go.Scatter(
                x=labels[:-1],
                y=[round(x * 100, 1) for x in base],
                mode="lines+markers",
                line={"color": "#6b7280", "width": 2, "dash": "dash"},
                marker={"size": 6, "color": "#6b7280"},
                name="Average Applicants",
            )
        )
        
        # Add current applicant point
        fig.add_trace(
            go.Scatter(
                x=[labels[-1]],
                y=[round(current_conf * 100, 1)],
                mode="markers",
                marker={"size": 16, "color": "#10b981", "line": {"color": "#fff", "width": 2}},
                name="Your Application",
            )
        )
        
        fig.update_layout(
            title={
                "text": "Confidence vs Typical Applicants",
                "font": {"color": "#e5e7eb", "size": 16}
            },
            xaxis={"title": "", "color": "#9ca3af", "gridcolor": "#1f2937"},
            yaxis={
                "title": "Confidence (%)",
                "color": "#9ca3af",
                "gridcolor": "#1f2937",
                "range": [0, 100]
            },
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(10,10,10,0.8)",
            margin=dict(l=50, r=20, t=60, b=40),
            height=300,
            showlegend=True,
            legend={"font": {"color": "#9ca3af"}},
        )
        
        return fig
    except Exception as e:
        logger.error(f"Error creating trend chart: {e}")
        return go.Figure()


# =========================================================
# PDF GENERATION
# =========================================================
def generate_pdf_summary(result: Dict[str, Any]) -> bytes:
    """Generate comprehensive PDF summary."""
    try:
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        
        # Title
        pdf.set_font("Arial", "B", 20)
        pdf.cell(0, 15, "Loan Approval Summary", ln=1, align="C")
        
        pdf.set_font("Arial", "", 11)
        pdf.ln(5)
        
        # Decision
        pdf.set_font("Arial", "B", 14)
        decision_text = f"Decision: {result['status']}"
        pdf.cell(0, 10, decision_text, ln=1)
        
        pdf.set_font("Arial", "", 12)
        pdf.cell(0, 8, f"Confidence Level: {result['confidence']:.1%}", ln=1)
        pdf.cell(0, 8, f"Date: {result['timestamp']}", ln=1)
        
        pdf.ln(5)
        
        data = result["input_data"]
        
        # Applicant Details
        pdf.set_font("Arial", "B", 13)
        pdf.cell(0, 10, "Applicant Details", ln=1)
        pdf.set_font("Arial", "", 11)
        
        pdf.cell(0, 7, f"Gender: {data['gender']}", ln=1)
        pdf.cell(0, 7, f"Marital Status: {data['married']}", ln=1)
        pdf.cell(0, 7, f"Number of Dependents: {data['dependents']}", ln=1)
        pdf.cell(0, 7, f"Education: {data['education']}", ln=1)
        
        emp_status = "Self Employed" if data["self_employed"] == "Yes" else "Not Self Employed"
        pdf.cell(0, 7, f"Employment Status: {emp_status}", ln=1)
        
        pdf.ln(5)
        
        # Financial Information
        pdf.set_font("Arial", "B", 13)
        pdf.cell(0, 10, "Financial Information", ln=1)
        pdf.set_font("Arial", "", 11)
        
        pdf.cell(0, 7, f"Monthly Income: Rs. {data['applicant_income']:,}", ln=1)
        pdf.cell(0, 7, f"Co-applicant Income: Rs. {data['coapplicant_income']:,}", ln=1)
        
        total_income = data['applicant_income'] + data['coapplicant_income']
        pdf.cell(0, 7, f"Total Monthly Income: Rs. {total_income:,}", ln=1)
        
        pdf.cell(0, 7, f"Loan Amount Requested: Rs. {data['loan_amount']:,}", ln=1)
        pdf.cell(0, 7, f"Loan Term: {data['loan_amount_term']} months", ln=1)
        
        credit_status = "Good" if data["credit_history"] == 1 else "Needs Improvement"
        pdf.cell(0, 7, f"Credit History: {credit_status}", ln=1)
        
        pdf.ln(5)
        
        # Property Details
        pdf.set_font("Arial", "B", 13)
        pdf.cell(0, 10, "Property Details", ln=1)
        pdf.set_font("Arial", "", 11)
        
        pdf.cell(0, 7, f"Property Area: {data['property_area']}", ln=1)
        
        pdf.ln(5)
        
        # Key Metrics
        pdf.set_font("Arial", "B", 13)
        pdf.cell(0, 10, "Key Metrics", ln=1)
        pdf.set_font("Arial", "", 11)
        
        if total_income > 0:
            dti = (data['loan_amount'] / total_income) * 100
            lti = (data['loan_amount'] / max(1, data['applicant_income'])) * 100
            pdf.cell(0, 7, f"Debt to Income Ratio: {dti:.1f}%", ln=1)
            pdf.cell(0, 7, f"Loan to Income Ratio: {lti:.1f}%", ln=1)
        
        pdf.ln(10)
        
        # Footer
        pdf.set_font("Arial", "I", 9)
        pdf.cell(0, 6, "This is a system-generated report.", ln=1)
        pdf.cell(0, 6, "Powered by AI Loan Approval System", ln=1)
        
        pdf_bytes = pdf.output(dest="S").encode("latin1")
        logger.info("PDF generated successfully")
        return pdf_bytes
    
    except Exception as e:
        logger.error(f"Error generating PDF: {e}")
        raise RuntimeError(f"PDF generation failed: {e}")


# =========================================================
# RESULT DISPLAY FUNCTION
# =========================================================
def display_result(result: Dict[str, Any], result_key: str = ""):
    """Display prediction result with detailed analysis."""
    if not result:
        return
    
    data = result["input_data"]
    
    # Add a subtle animation effect
    st.balloons()
    
    # Decision Banner with enhanced styling
    st.markdown("<div style='animation: fadeIn 0.5s ease-in;'>", unsafe_allow_html=True)
    
    if result["status"] == "Approved":
        st.markdown(
            """
        <div class="status-approved">
            <div style="font-size:4rem;margin-bottom:0.75rem;position:relative;z-index:1;">✅</div>
            <h1 style="margin:0;font-size:2.5rem;font-weight:700;position:relative;z-index:1;">
                Loan Approved
            </h1>
            <p style="margin:0.75rem 0 0;font-size:1.15rem;opacity:0.95;position:relative;z-index:1;">
                Congratulations! Your application meets our underwriting criteria.
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
        <div class="status-rejected">
            <div style="font-size:4rem;margin-bottom:0.75rem;">❌</div>
            <h1 style="margin:0;font-size:2.5rem;font-weight:700;">
                Application Not Approved
            </h1>
            <p style="margin:0.75rem 0 0;font-size:1.15rem;opacity:0.95;">
                Your current profile doesn't meet our risk criteria. See recommendations below.
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)  # Close animation div

    # Key Metrics Row
    st.markdown("### 📊 Key Metrics Dashboard")
    st.markdown("<div style='animation: fadeIn 0.6s ease-in 0.1s both;'>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1.5, 1.5, 1])

    applicant_income = data["applicant_income"]
    co_income = data["coapplicant_income"]
    total_income = max(1, applicant_income + co_income)
    debt_to_income = (data["loan_amount"] / total_income) * 100
    loan_to_income = (data["loan_amount"] / max(1, applicant_income)) * 100

    with col1:
        st.markdown(
            f"""
        <div class="card">
            <div style="margin-bottom:1.5rem;">
                <div style="font-size:0.85rem;color:#9ca3af;margin-bottom:0.25rem;">Decision Timestamp</div>
                <div style="font-size:1.15rem;color:#e5e7eb;font-weight:600;">{result['timestamp']}</div>
            </div>
            <div>
                <div style="font-size:0.85rem;color:#9ca3af;margin-bottom:0.25rem;">Combined Monthly Income</div>
                <div style="font-size:1.6rem;color:#10b981;font-weight:700;">₹{total_income:,}</div>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.plotly_chart(approval_gauge(result["confidence"]), use_container_width=True, key=f"gauge_{result_key}")

    with col3:
        dti_color = "#10b981" if debt_to_income < 40 else "#f59e0b" if debt_to_income < 60 else "#ef4444"
        st.markdown(
            f"""
        <div class="card">
            <div style="margin-bottom:1rem;">
                <div style="font-size:0.9rem;color:#9ca3af;margin-bottom:0.25rem;">Debt to Income</div>
                <div style="font-size:1.8rem;font-weight:700;color:{dti_color};">{debt_to_income:.1f}%</div>
            </div>
            <div>
                <div style="font-size:0.85rem;color:#9ca3af;margin-bottom:0.25rem;">Loan to Income</div>
                <div style="font-size:1.3rem;font-weight:600;color:#e5e7eb;">{loan_to_income:.1f}%</div>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)  # Close metrics animation
    
    # AI Recommendations Section
    st.markdown("### 🤖 AI Credit Officer Analysis")
    st.markdown("<div style='animation: fadeIn 0.8s ease-in 0.3s both;'>", unsafe_allow_html=True)

    if has_valid_openrouter_key():
        col_rec1, col_rec2 = st.columns([3, 1])
        
        with col_rec1:
            if st.button(f"Generate AI Recommendations {result_key}", use_container_width=True, type="primary", key=f"ai_rec_{result_key}"):
                with st.spinner("🔄 AI is analyzing your profile and generating recommendations..."):
                    try:
                        explanation = ai_recommendations(data, result)
                        st.session_state[f"ai_recommendations_{result_key}"] = explanation
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error generating recommendations: {str(e)}")
                        logger.error(f"AI recommendations error: {e}")
        
        with col_rec2:
            if st.session_state.get(f"ai_recommendations_{result_key}"):
                if st.button(f"Clear Analysis {result_key}", use_container_width=True, key=f"clear_ai_{result_key}"):
                    st.session_state[f"ai_recommendations_{result_key}"] = None
                    st.rerun()
        
        if st.session_state.get(f"ai_recommendations_{result_key}"):
            st.markdown(
                f"""
            <div class="card" style="font-size:0.975rem;color:#e5e7eb;line-height:1.8;">
                {st.session_state[f'ai_recommendations_{result_key}']}
            </div>
            """,
                unsafe_allow_html=True,
            )
    else:
        st.info("🔑 Configure `OPENROUTER_API_KEY` in your secrets to enable AI-powered recommendations and insights.")

    st.markdown("</div>", unsafe_allow_html=True)  # Close AI recommendations animation
    
    # Static Recommendations
    st.markdown("### 🎯 Risk Model Insights")
    st.markdown("<div style='animation: fadeIn 0.9s ease-in 0.4s both;'>", unsafe_allow_html=True)

    if result["status"] == "Approved":
        recs = [
            ("✅", "Strong Application Profile", "Your application demonstrates solid financial stability and creditworthiness."),
            ("📊", "Income Sufficiency", "Your income adequately supports the requested loan amount and tenure."),
            ("⭐", "Credit Standing", "Good credit history strengthens your application significantly."),
            ("💡", "Maintain Good Practices", "Continue timely EMI payments and maintain low credit utilization."),
        ]
    else:
        recs = [
            ("📉", "Reduce Loan Amount", "Consider requesting a lower amount (20-30% reduction) to improve approval odds."),
            ("⏰", "Extend Loan Tenure", "Longer tenure reduces monthly EMI burden and improves debt-to-income ratio."),
            ("💰", "Increase Income Documentation", "Add co-applicant or demonstrate additional income sources."),
            ("⭐", "Build Credit History", "Work on improving credit score through timely bill payments and low credit utilization."),
            ("📋", "Reduce Existing Debts", "Pay off or reduce existing loans before reapplying for better chances."),
        ]

    for icon, title, description in recs:
        st.markdown(
            f"""
        <div style='display:flex;align-items:flex-start;gap:1rem;padding:1rem 1.25rem;
                    background:linear-gradient(135deg, #0f172a 0%, #0a0a0a 100%);
                    border-radius:16px;margin-bottom:0.75rem;
                    border:1px solid rgba(59, 130, 246, 0.3);
                    box-shadow:0 4px 12px rgba(0, 0, 0, 0.5);'>
            <div style='font-size:1.5rem;flex-shrink:0;'>{icon}</div>
            <div style='flex:1;'>
                <div style='font-size:1rem;color:#e5e7eb;font-weight:600;margin-bottom:0.25rem;'>
                    {title}
                </div>
                <div style='font-size:0.9rem;color:#9ca3af;line-height:1.6;'>
                    {description}
                </div>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("</div>", unsafe_allow_html=True)  # Close insights animation
    
    # Action Buttons
    st.markdown("### ⚡ Quick Actions")
    st.markdown("<div style='animation: fadeIn 1s ease-in 0.5s both;'>", unsafe_allow_html=True)

    col_act1, col_act2, col_act3 = st.columns(3, gap="medium")

    with col_act1:
        if st.button(f"💾 Save as JSON {result_key}", use_container_width=True, key=f"save_json_{result_key}"):
            try:
                ts = result["timestamp"].replace(":", "-").replace(" ", "_")
                fname = f"loan_application_{ts}_{result_key}.json"
                with open(fname, "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                st.success(f"✅ Saved as {fname}")
                logger.info(f"Application saved as JSON: {fname}")
            except Exception as e:
                st.error(f"❌ Error saving file: {str(e)}")
                logger.error(f"JSON save error: {e}")

    with col_act2:
        if st.button(f"📄 Generate PDF {result_key}", use_container_width=True, key=f"gen_pdf_{result_key}"):
            with st.spinner("📝 Generating PDF report..."):
                try:
                    pdf_bytes = generate_pdf_summary(result)
                    st.session_state[f"pdf_bytes_{result_key}"] = pdf_bytes
                    st.success("✅ PDF generated successfully!")
                except Exception as e:
                    st.error(f"❌ Error generating PDF: {str(e)}")
                    logger.error(f"PDF generation error: {e}")

        if st.session_state.get(f"pdf_bytes_{result_key}"):
            st.download_button(
                f"⬇️ Download PDF {result_key}",
                data=st.session_state[f"pdf_bytes_{result_key}"],
                file_name=f"loan_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{result_key}.pdf",
                mime="application/pdf",
                use_container_width=True,
                key=f"download_pdf_{result_key}"
            )

    with col_act3:
        if st.button(f"🔄 New Analysis {result_key}", use_container_width=True, key=f"new_analysis_{result_key}"):
            if result_key == "structured":
                st.session_state.structured_result = None
            elif result_key == "ai":
                st.session_state.ai_result = None
                st.session_state.ai_extracted_data = None
            st.session_state[f"ai_recommendations_{result_key}"] = None
            st.session_state[f"pdf_bytes_{result_key}"] = None
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)  # Close actions animation


# =========================================================
# MAIN APPLICATION HEADER
# =========================================================
st.markdown(
    """
<div class="main-header">
    <div style="display:flex;align-items:center;justify-content:space-between;gap:1.5rem;position:relative;z-index:1;">
        <div style="display:flex;align-items:center;gap:1.5rem;">
            <div style="font-size:3rem;filter:drop-shadow(0 4px 12px rgba(59, 130, 246, 0.6));">🤖</div>
            <div>
                <h1 style="color:#ffffff;margin:0;font-size:2.8rem;font-weight:700;
                           text-shadow:0 2px 20px rgba(59, 130, 246, 0.5);">
                    AI Loan Approval Studio
                </h1>
                <p style="color:#9ca3af;margin:0.5rem 0 0;font-size:1.05rem;font-weight:400;">
                    Enterprise-grade underwriting platform powered by machine learning and OpenRouter AI
                </p>
            </div>
        </div>
        <div style="display:flex;flex-direction:column;gap:0.6rem;align-items:flex-end;">
            <div class="metric-pill">🇮🇳 India • Retail Lending • Real-time</div>
            <div class="metric-pill">⚡ Model v2.1 • UI v3.0 Pro</div>
        </div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

# =========================================================
# MAIN TABS
# =========================================================
tab_names = ["📝 Structured Form", "💬 AI Natural Language"]

# Custom tab implementation with session state
selected_tab = st.radio(
    "Navigation",
    range(len(tab_names)),
    format_func=lambda x: tab_names[x],
    key="tab_selector",
    horizontal=True,
    label_visibility="collapsed",
    index=st.session_state.active_tab
)

# Update active tab
st.session_state.active_tab = selected_tab

# Custom styling for radio buttons to look like tabs
st.markdown("""
<style>
    div[data-testid="stRadio"] {
        background: transparent;
        padding: 0;
    }
    div[data-testid="stRadio"] > div {
        display: flex;
        gap: 0.5rem;
        border-bottom: 2px solid rgba(59, 130, 246, 0.2);
        padding-bottom: 0;
        margin-bottom: 2rem;
    }
    div[data-testid="stRadio"] > div > label {
        background: rgba(15, 23, 42, 0.5);
        padding: 1rem 2rem;
        border-radius: 16px 16px 0 0;
        border: 1px solid transparent;
        color: #9ca3af;
        cursor: pointer;
        transition: all 0.3s ease;
        font-weight: 500;
        font-size: 1rem;
    }
    div[data-testid="stRadio"] > div > label:hover {
        background: rgba(15, 23, 42, 0.8);
        color: #e5e7eb;
    }
    div[data-testid="stRadio"] > div > label[data-checked="true"] {
        background: linear-gradient(135deg, #1e40af 0%, #4f46e5 100%) !important;
        color: #ffffff !important;
        border-color: rgba(59, 130, 246, 0.8) !important;
        box-shadow: 0 8px 24px rgba(59, 130, 246, 0.4);
    }
    div[data-testid="stRadio"] input[type="radio"] {
        display: none;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================
# TAB 1: STRUCTURED FORM
# =========================================================
if selected_tab == 0:
    st.markdown("### 📋 Loan Application Form")
    st.markdown("##### Fill in the details below to get instant approval decision")

    col_left, col_right = st.columns([2.5, 1.5], gap="large")

    with col_left:
        with st.form("loan_form", clear_on_submit=False):
            col_a, col_b = st.columns(2, gap="medium")

            with col_a:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("#### Personal Information")
                
                gender = st.selectbox("Gender", ["Male", "Female"], key="gender")
                married = st.selectbox("Marital Status", ["Yes", "No"], key="married")
                dependents = st.selectbox("Number of Dependents", ["0", "1", "2", "3+"], key="dependents")
                education = st.selectbox("Education Level", ["Graduate", "Not Graduate"], key="education")
                self_employed = st.selectbox("Employment Type", ["No", "Yes"], 
                                             format_func=lambda x: "Self Employed" if x == "Yes" else "Salaried",
                                             key="self_employed")
                
                st.markdown("</div>", unsafe_allow_html=True)

            with col_b:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.markdown("#### Financial Details")
                
                applicant_income = st.number_input(
                    "Monthly Income (₹)", 
                    min_value=0, 
                    value=45000, 
                    step=5000,
                    help="Your monthly income in Indian Rupees",
                    key="applicant_income"
                )
                
                coapplicant_income = st.number_input(
                    "Co-applicant Income (₹)", 
                    min_value=0, 
                    value=0, 
                    step=5000,
                    help="Co-applicant's monthly income (if applicable)",
                    key="coapplicant_income"
                )
                
                loan_amount = st.number_input(
                    "Loan Amount (₹)", 
                    min_value=10000, 
                    value=500000, 
                    step=10000,
                    help="Total loan amount requested",
                    key="loan_amount"
                )
                
                loan_term = st.selectbox(
                    "Loan Term (months)",
                    [360, 300, 240, 180, 120, 84, 60, 36, 24, 12],
                    index=2,
                    help="Loan repayment period in months",
                    key="loan_term"
                )
                
                credit_history = st.selectbox(
                    "Credit History",
                    [1, 0],
                    format_func=lambda x: "✅ Good (1)" if x == 1 else "⚠️ Poor (0)",
                    help="1 = Good credit history, 0 = Poor/No credit history",
                    key="credit_history"
                )
                
                property_area = st.selectbox(
                    "Property Area",
                    ["Urban", "Semiurban", "Rural"],
                    help="Location of the property",
                    key="property_area"
                )
                
                st.markdown("</div>", unsafe_allow_html=True)

            # Store form data
            form_data = {
                "gender": gender,
                "married": married,
                "dependents": dependents,
                "education": education,
                "self_employed": self_employed,
                "applicant_income": applicant_income,
                "coapplicant_income": coapplicant_income,
                "loan_amount": loan_amount,
                "loan_amount_term": loan_term,
                "credit_history": credit_history,
                "property_area": property_area,
            }

            st.markdown("<br>", unsafe_allow_html=True)
            
            col_submit, col_clear = st.columns([3, 1])

            with col_submit:
                submitted = st.form_submit_button(
                    "🚀 Analyze Application",
                    type="primary",
                    use_container_width=True
                )

            with col_clear:
                clear_pressed = st.form_submit_button(
                    "🗑️ Clear",
                    use_container_width=True
                )

            if clear_pressed:
                st.session_state.structured_result = None
                st.session_state.ai_recommendations_structured = None
                st.session_state.pdf_bytes_structured = None
                st.rerun()

            if submitted:
                with st.spinner("🔍 Running ML risk assessment..."):
                    try:
                        time.sleep(0.5)
                        result = make_prediction(form_data)
                        st.session_state.structured_result = result
                        st.success("✅ Analysis complete!")
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                        logger.error(f"Prediction error: {e}")

    with col_right:
        st.markdown("### 💡 Expert Tips")
        
        tips = [
            ("📈", "Maintain DTI Ratio Below 40%", "Keep your total debt-to-income ratio under 40% for better approval chances."),
            ("⭐", "Build Strong Credit History", "A credit score above 750 significantly improves your approval odds."),
            ("💰", "Stable Income Source", "Consistent income history of 2+ years strengthens your application."),
            ("👥", "Consider Co-applicant", "Adding a co-applicant with good income can boost approval probability."),
            ("🏠", "Property Location Matters", "Urban properties often get better loan terms and higher amounts."),
            ("📊", "Loan-to-Value Ratio", "Keep LTV below 80% for residential properties for optimal terms."),
        ]

        for icon, title, description in tips:
            st.markdown(
                f"""
            <div class="card" style="padding:1rem 1.25rem;margin-bottom:0.75rem;">
                <div style="display:flex;align-items:flex-start;gap:1rem;">
                    <div style="font-size:1.5rem;flex-shrink:0;">{icon}</div>
                    <div>
                        <div style="font-size:1rem;color:#e5e7eb;font-weight:600;margin-bottom:0.25rem;">
                            {title}
                        </div>
                        <div style="font-size:0.875rem;color:#9ca3af;line-height:1.5;">
                            {description}
                        </div>
                    </div>
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

        if not has_valid_openrouter_key():
            st.info(
                "💡 **Pro Tip:** Add `OPENROUTER_API_KEY` to `.streamlit/secrets.toml` "
                "to enable AI-powered natural language processing and recommendations."
            )

    # Display structured form results if available
    if st.session_state.structured_result:
        st.markdown('<div class="results-section">', unsafe_allow_html=True)
        st.markdown("### 🎯 Structured Form Analysis Results")
        display_result(st.session_state.structured_result, "structured")
        st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# TAB 2: AI NATURAL LANGUAGE
# =========================================================
else:
    st.markdown("### 🤖 Describe Your Loan Requirements")
    st.markdown("##### Our AI will extract all necessary information from your description")

    col_input, col_output = st.columns([3, 2], gap="large")

    with col_input:
        st.markdown(
            """
        <div class="card">
            <h4 style="margin-top:0;color:#e5e7eb;">📝 Example Description</h4>
            <p style="color:#9ca3af;font-size:0.95rem;line-height:1.6;font-style:italic;">
                "I'm a 32-year-old software engineer working in Bangalore. I earn ₹95,000 per month 
                and my spouse earns ₹40,000. We're looking for a home loan of ₹35 lakhs for 20 years. 
                We have excellent credit history with a score of 780, one child, and we're looking at 
                a property in an urban area. No existing EMIs."
            </p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        user_input = st.text_area(
            "Your Application Story:",
            height=180,
            placeholder="Describe your situation: income, family details, loan requirements, credit history, property details, etc...",
            key="natural_input",
            help="Be as detailed as possible for accurate analysis"
        )

        col_btn1, col_btn2 = st.columns([2, 1])

        with col_btn1:
            analyze_clicked = st.button(
                "🤖 Analyze with AI",
                use_container_width=True,
                type="primary"
            )

        with col_btn2:
            clear_ai = st.button("🗑️ Clear", use_container_width=True)

        if clear_ai:
            st.session_state.ai_result = None
            st.session_state.ai_extracted_data = None
            st.session_state.ai_recommendations_ai = None
            st.session_state.pdf_bytes_ai = None
            st.rerun()

        if analyze_clicked:
            if not user_input.strip():
                st.warning("⚠️ Please enter your loan requirements.")
            elif not has_valid_openrouter_key():
                st.error("❌ OpenRouter API key not configured. Please add it to your secrets.")
            else:
                with st.spinner("🔄 AI is analyzing your description..."):
                    try:
                        extracted = ai_extract_features_from_text(user_input)
                        st.session_state.ai_extracted_data = extracted
                        
                        with st.spinner("🔍 Running risk assessment..."):
                            result = make_prediction(extracted)
                            st.session_state.ai_result = result
                        
                        st.success("✅ AI analysis complete!")
                    except Exception as e:
                        st.error(f"❌ Error during AI analysis: {str(e)}")
                        logger.error(f"AI extraction error: {e}")

    with col_output:
        st.markdown("### 🎯 Extracted Features")

        info = st.session_state.ai_extracted_data
        if info:
            st.markdown('<div class="card" style="font-size:0.95rem;">', unsafe_allow_html=True)
            st.markdown("<h4 style='margin-top:0;color:#10b981;'>✅ AI-Extracted Profile</h4>", unsafe_allow_html=True)

            # Create a nice formatted display
            feature_display = {
                "👤 Personal": {
                    "Gender": info['gender'],
                    "Married": info['married'],
                    "Dependents": info['dependents'],
                    "Education": info['education'],
                    "Employment": "Self Employed" if info['self_employed'] == "Yes" else "Salaried"
                },
                "💰 Financial": {
                    "Monthly Income": f"₹{info['applicant_income']:,}",
                    "Co-applicant Income": f"₹{info['coapplicant_income']:,}",
                    "Loan Amount": f"₹{info['loan_amount']:,}",
                    "Loan Term": f"{info['loan_amount_term']} months",
                    "Credit History": "Good" if info['credit_history'] == 1 else "Poor"
                },
                "🏠 Property": {
                    "Area Type": info['property_area']
                }
            }

            for section, fields in feature_display.items():
                st.markdown(f"<div style='margin:1rem 0;'><strong style='color:#3b82f6;font-size:1.05rem;'>{section}</strong></div>", unsafe_allow_html=True)
                for key, value in fields.items():
                    st.markdown(
                        f"""
                        <div style='display:flex;justify-content:space-between;padding:0.5rem 0;
                                   border-bottom:1px dashed rgba(59, 130, 246, 0.2);'>
                            <span style='color:#9ca3af;'>{key}:</span>
                            <span style='color:#e5e7eb;font-weight:600;'>{value}</span>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.info("💭 Run the AI analyzer to see extracted features here.")

    # Display AI analysis results if available
    if st.session_state.ai_result:
        st.markdown('<div class="results-section">', unsafe_allow_html=True)
        st.markdown("### 🤖 AI Analysis Results")
        display_result(st.session_state.ai_result, "ai")
        st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
st.markdown(
    """
<div style='text-align:center;color:#6b7280;font-size:0.9rem;padding:1.5rem;
           background:linear-gradient(135deg, rgba(10, 10, 10, 0.8) 0%, rgba(15, 23, 42, 0.5) 100%);
           border-radius:16px;margin-top:2rem;'>
    <div style='display:flex;justify-content:center;gap:2rem;flex-wrap:wrap;margin-bottom:1rem;'>
        <span style='display:flex;align-items:center;gap:0.5rem;'>
            <span style='font-size:1.2rem;'>🔒</span> Enterprise Security
        </span>
        <span style='display:flex;align-items:center;gap:0.5rem;'>
            <span style='font-size:1.2rem;'>🤖</span> OpenRouter AI
        </span>
        <span style='display:flex;align-items:center;gap:0.5rem;'>
            <span style='font-size:1.2rem;'>⚡</span> Real-time Analysis
        </span>
        <span style='display:flex;align-items:center;gap:0.5rem;'>
            <span style='font-size:1.2rem;'>📊</span> ML-Powered
        </span>
    </div>
    <div style='font-size:0.85rem;color:#9ca3af;'>
        © 2026 AI Loan Approval Studio • Production-Grade Fintech Solution
    </div>
    <div style='margin-top:0.75rem;font-size:0.8rem;color:#6b7280;'>
        Built with Streamlit • Python • Machine Learning • OpenRouter API
    </div>
</div>
""",
    unsafe_allow_html=True,
)
