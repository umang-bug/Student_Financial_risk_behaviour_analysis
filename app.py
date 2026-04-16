import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from scipy.spatial.distance import cdist
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Student Financial Profiler", layout="wide")

# --- CUSTOM CSS FOR DESIGN ---
st.markdown("""
    <style>
    .main { background: #0e1117; color: #ffffff; }
    .main-title { 
        font-size: 3.5rem; font-weight: 800; text-align: center;
        background: -webkit-linear-gradient(#4facfe, #00f2fe);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .stButton>button { 
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%); 
        color: black; font-weight: bold; border-radius: 50px; border: none;
    }
    .metric-card { 
        background: #161b22; padding: 25px; border-radius: 15px; 
        border: 1px solid #30363d; text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD MODELS ---
@st.cache_resource
def load_artifacts():
    art = {}
    try:
        # Loading all 5 required files from your GitHub models/ folder
        art['scaler'] = joblib.load('models/scaler.joblib')
        art['encoder'] = joblib.load('models/ordinal_encoder.joblib')
        art['kp'] = joblib.load('models/kproto_model.joblib')
        art['rf'] = joblib.load('models/risk_rf_model.joblib')
        art['nn'] = tf.keras.models.load_model('models/spend_nn_model.h5')
        
        # Optional distance logic files
        art['feat_imp'] = joblib.load('models/feature_importances.joblib') if os.path.exists('models/feature_importances.joblib') else None
        art['safe_ref'] = joblib.load('models/safe_reference.joblib') if os.path.exists('models/safe_reference.joblib') else None
    except Exception as e:
        st.error(f"Error loading models: {e}")
    return art

if 'page' not in st.session_state: 
    st.session_state.page = "Home"

def main():
    # --- 1. HOMEPAGE ---
    if st.session_state.page == "Home":
        st.markdown("<div class='main-title'>Financial Intelligence System</div>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #8b949e;'>Behavioral Risk Analysis using Neural Networks & Clustering</p>", unsafe_allow_html=True)
        st.markdown("<h1 style='text-align: center; font-size: 80px;'>🧠📊</h1>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1,1,1])
        if col2.button("Begin Assessment"):
            st.session_state.page = "Registration"
            st.rerun()

    # --- 2. REGISTRATION ---
    elif st.session_state.page == "Registration":
        st.header("👤 User Registration")
        with st.form("reg"):
            name = st.text_input("Enter your full name")
            age = st.number_input("Age", 17, 30, 20)
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            if st.form_submit_button("Proceed to Survey"):
                st.session_state.user_name = name
                st.session_state.page = "Survey"
                st.rerun()

    # --- 3. THE SURVEY ---
    elif st.session_state.page == "Survey":
        st.header(f"Assessment for {st.session_state.user_name}")
        art = load_artifacts()

        with st.form("ml_survey"):
            c1, c2 = st.columns(2)
            with c1:
                place = st.selectbox("Where did you grow up?", ["🏙️ Big metro city", "🏢 Medium-sized city", "🏘️ Small town", "🌾 Rural area"])
                track = st.selectbox("How do you track monthly expenditures?", ["I check my bank balance occasionally.", "I review my history within payment apps (e.g., UPI, Paytm).", "I do not keep the track", "I use a dedicated expense-tracking app or spreadsheet."])
                graph = st.selectbox("Select your spending pattern graph:", ["Uniform Daily Expenses", "Irregular and Random Spending", "Spend a lot once and then low spending for rest", "Steady Weekdays with High Weekends"])
            
            with c2:
                unplanned = st.slider("Unplanned purchase frequency (1-5)", 1, 5)
                confidence = st.slider("Financial management confidence (1-5)", 1, 5)
                peer_infl = st.slider("Peer pressure influence (1-5)", 1, 5)

            st.write("---")
            st.write("**Factor Importance (1: Low, 3: High)**")
            sc1, sc2, sc3, sc4 = st.columns(4)
            p_price = sc1.select_slider("Price", ["Not important", "Slightly important", "Very important"])
            p_brand = sc2.select_slider("Brand", ["Not important", "Slightly important", "Very important"])
            p_peer = sc3.select_slider("Peer Rec", ["Not important", "Slightly important", "Very important"])
            p_util = sc4.select_slider("Utility", ["Not important", "Slightly important", "Very important"])

            st.write("---")
            st.write("**Budget Breakdown (Check all that apply)**")
            bc1, bc2, bc3, bc4, bc5 = st.columns(5)
            b_food = bc1.checkbox("Food")
            b_travel = bc2.checkbox("Travel")
            b_fashion = bc3.checkbox("Fashion")
            b_sub = bc4.checkbox("Subscriptions")
            b_fun = bc5.checkbox("Entertainment")

            if st.form_submit_button("RUN ML ANALYSIS"):
                # --- PREPROCESSING ---
                # 1. Categorical Encoding (5 values)
                cat_row = [[place, p_price, p_brand, p_peer, p_util]]
                enc_cats = art['encoder'].transform(cat_row)[0]
                
                # 2. Numeric Scaling (3 values)
                enc_nums = art['scaler'].transform([[unplanned, peer_infl, confidence]])[0]
                
                # 3. COMPILING FEATURES (13 TOTAL)
                # Matches your broken-out budget categories logic
                features_list = list(enc_cats) + list(enc_nums) + [
                    int(b_food), int(b_travel), int(b_fashion), int(b_sub), int(b_fun)
                ]
                
                features = np.array([features_list])

                # --- PREDICTIONS ---
                # NN Spend Prediction
                nn_out = art['nn'].predict(features, verbose=0)
                spend_pred = np.argmax(nn_out) + 1
                
                # Clustering Prediction
                cluster_id = art['kp'].predict(features, categorical=[0,1,2,3,4])[0]
                
                # Risk Score (Distance fallback)
                risk_score = 48.0
                if art['safe_ref'] is not None and art['feat_imp'] is not None:
                    try:
                        w_feat = features[0][:len(art['feat_imp'])] * art['feat_imp']
                        w_safe = art['safe_ref'][:len(art['feat_imp'])] * art['feat_imp']
                        risk_score = min(100, cdist([w_feat], [w_safe], metric='euclidean')[0][0] * 15)
                    except: pass

                st.session_state.results = {'spend': spend_pred, 'cluster': cluster_id, 'risk': risk_score}
                st.session_state.page = "Dashboard"
                st.rerun()

    # --- 4. DASHBOARD ---
    elif st.session_state.page == "Dashboard":
        st.markdown("<div class='main-title'>Behavioral Report</div>", unsafe_allow_html=True)
        res = st.session_state.results
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"<div class='metric-card'><h3>Spend Tier</h3><h2 style='color:#4facfe'>Tier {res['spend']}</h2></div>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<div class='metric-card'><h3>Behavior Cluster</h3><h2 style='color:#00f2fe'>Group {res['cluster']}</h2></div>", unsafe_allow_html=True)
        with col3:
            st.markdown(f"<div class='metric-card'><h3>Risk Score</h3><h2 style='color:#ff4b4b'>{res['risk']:.1f}/100</h2></div>", unsafe_allow_html=True)

        if st.button("Restart Analysis"):
            st.session_state.page = "Home"
            st.rerun()

if __name__ == "__main__":
    main()
