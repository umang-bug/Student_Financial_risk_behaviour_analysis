import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import requests
from scipy.spatial.distance import cdist

# --- PAGE CONFIG ---
st.set_page_config(page_title="Financial Risk Profiler", layout="wide", initial_sidebar_state="collapsed")

# --- CUSTOM CSS FOR ML DESIGN ---
st.markdown("""
    <style>
    .main { background: radial-gradient(circle, #1a1c24 0%, #0e1117 100%); color: white; }
    .stButton>button { 
        width: 100%; border-radius: 10px; height: 3.5em; 
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%); 
        color: black; font-weight: bold; border: none;
    }
    .metric-card { 
        background-color: rgba(255, 255, 255, 0.05); 
        padding: 20px; border-radius: 15px; 
        border: 1px solid rgba(255, 255, 255, 0.1);
        text-align: center;
    }
    .title-text {
        background: -webkit-linear-gradient(#4facfe, #00f2fe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem; font-weight: 800; text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD MODELS SAFELY ---
@st.cache_resource
def load_models():
    # Matching filenames exactly from your GitHub
    scaler = joblib.load('models/scaler.joblib')
    # Using your uploaded Ordinal Encoder
    encoder = joblib.load('models/ordinal_encoder.joblib') 
    kp_model = joblib.load('models/kproto_model.joblib')
    rf_model = joblib.load('models/risk_rf_model.joblib')
    nn_model = tf.keras.models.load_model('models/spend_nn_model.h5')
    
    # Optional logic for distance-based risk scoring
    try:
        safe_ref = joblib.load('models/safe_reference.joblib')
        feat_imp = joblib.load('models/feature_importances.joblib')
    except:
        safe_ref, feat_imp = None, None
        
    return scaler, encoder, kp_model, rf_model, nn_model, safe_ref, feat_imp

# Initialize Session State
if 'step' not in st.session_state: st.session_state.step = "Home"

def main():
    # --- 1. PROFESSIONAL HOMEPAGE ---
    if st.session_state.step == "Home":
        st.markdown("<div class='title-text'>STUDENT FINANCIAL INTELLIGENCE</div>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; font-size: 1.2rem; color: #8899ac;'>Advanced Behavioral Analysis & Risk Profiling via Machine Learning</p>", unsafe_allow_html=True)
        
        # Illustration placeholder using standard markdown/emojis for reliability
        st.markdown("<h1 style='text-align: center; font-size: 100px;'>🧠⚡📊</h1>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1,1,1])
        with col2:
            st.write("---")
            st.info("Our model uses Neural Networks and K-Prototypes clustering to analyze your spending DNA.")
            if st.button("INITIATE PROFILING"):
                st.session_state.step = "User_Info"
                st.rerun()

    # --- 2. USER REGISTRATION ---
    elif st.session_state.step == "User_Info":
        st.header("👤 Identity Registration")
        with st.form("user_details"):
            name = st.text_input("Enter your full name")
            age = st.number_input("Age", 17, 30, 20)
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            if st.form_submit_button("Proceed to Assessment"):
                st.session_state.user_name = name
                st.session_state.step = "Survey"
                st.rerun()

    # --- 3. THE SURVEY ---
    elif st.session_state.step == "Survey":
        st.header(f"Financial Habits Assessment for {st.session_state.user_name}")
        scaler, encoder, kp_model, rf_model, nn_model, safe_ref, feat_imp = load_models()

        with st.form("ml_survey"):
            col1, col2 = st.columns(2)
            # Questions based on your survey sheet
            with col1:
                place = st.selectbox("Place you grew up in?", ["🏙️ Big metro city", "🏢 Medium-sized city", "🏘️ Small town", "🌾 Rural area"])
                track = st.selectbox("Expenditure tracking method?", ["I check my bank balance occasionally.", "I review my history within payment apps (e.g., UPI, Paytm).", "I do not keep the track", "I use a dedicated expense-tracking app or spreadsheet."])
                graph = st.selectbox("Expected spending graph?", ["Uniform Daily Expenses", "Irregular and Random Spending", "Spend a lot once and then low spending for rest", "Steady Weekdays with High Weekends"])

            with col2:
                unplanned = st.slider("Unplanned purchase frequency (1-5)", 1, 5)
                confidence = st.slider("Financial management confidence (1-5)", 1, 5)
                peer_infl = st.slider("Peer pressure influence (1-5)", 1, 5)

            st.write("**Purchase Factor Importance**")
            c1, c2, c3, c4 = st.columns(4)
            p_price = c1.select_slider("Price", ["Not important", "Slightly important", "Very important"])
            p_brand = c2.select_slider("Brand", ["Not important", "Slightly important", "Very important"])
            p_peer = c3.select_slider("Peer Rec", ["Not important", "Slightly important", "Very important"])
            p_util = c4.select_slider("Utility", ["Not important", "Slightly important", "Very important"])

            st.write("**Budget Allocation**")
            b_cols = st.columns(5)
            b_food = b_cols[0].checkbox("Food")
            b_travel = b_cols[1].checkbox("Travel")
            b_fashion = b_cols[2].checkbox("Fashion")
            b_sub = b_cols[3].checkbox("Subscriptions")
            b_fun = b_cols[4].checkbox("Entertainment")

            if st.form_submit_button("RUN ANALYSIS"):
                # PREPROCESSING
                # Creating a row that matches the Ordinal Encoder input
                raw_cats = [[place, p_price, p_brand, p_peer, p_util]]
                encoded_cats = encoder.transform(raw_cats)[0]
                
                # Numeric Scaling
                scaled_nums = scaler.transform([[unplanned, peer_infl, confidence]])[0]
                
                # Feature Vector Construction
                features = np.array([list(encoded_cats) + list(scaled_nums) + [int(b_food), int(b_travel), int(b_fashion), int(b_sub), int(b_fun)]])
                
                # PREDICTIONS
                # NN Softmax for Spend Tier
                spend_pred = np.argmax(nn_model.predict(features)) + 1
                
                # K-Proto for Cluster
                # Categorical indices must match your training (typically 0-4 for the encoded strings)
                cluster_id = kp_model.predict(features, categorical=[0,1,2,3,4])[0]
                
                # Risk Score (Distance Logic)
                # Risk Score (Distance Logic)
                if safe_ref is not None:
                    weighted_feat = features[0][:len(feat_imp)] * feat_imp
                    weighted_safe = safe_ref[:len(feat_imp)] * feat_imp
                    risk_val = cdist([weighted_feat], [weighted_safe], metric='euclidean')[0][0]
                    # This line below was missing the closing parenthesis ')'
                    risk_score = min(100, risk_val * 15) 
                else:
                    risk_score = 50.0 # Default fallback
