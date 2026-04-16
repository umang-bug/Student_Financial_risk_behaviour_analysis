import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from streamlit_lottie import st_lottie
import requests
from scipy.spatial.distance import cdist

# --- PAGE CONFIG ---
st.set_page_config(page_title="Student Financial Profiler", layout="wide", initial_sidebar_state="collapsed")

# --- CUSTOM CSS FOR PROFESSIONAL LOOK ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: white; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #262730; color: white; }
    .stMetric { background-color: #161b22; padding: 15px; border-radius: 10px; border: 1px solid #30363d; }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD ASSETS SAFELY ---
def load_lottieurl(url):
    try:
        r = requests.get(url, timeout=5)
        return r.json() if r.status_code == 200 else None
    except: return None

lottie_sphere = load_lottieurl("https://lottie.host/8636f333-3330-4e33-875c-352b2f67644d/I9XGZl2Y1O.json")

# --- LOAD MODELS ---
@st.cache_resource
def load_models():
    scaler = joblib.load('models/scaler.joblib')
    encoders = joblib.load('models/label_encoders.joblib')
    kp_model = joblib.load('models/kproto_model.joblib')
    rf_model = joblib.load('models/risk_rf_model.joblib')
    nn_model = tf.keras.models.load_model('models/spend_nn_model.h5')
    safe_ref = joblib.load('models/safe_reference.joblib')
    feat_imp = joblib.load('models/feature_importances.joblib')
    return scaler, encoders, kp_model, rf_model, nn_model, safe_ref, feat_imp

# Initialize Session States
if 'step' not in st.session_state: st.session_state.step = "Home"

# --- MAIN APP LOGIC ---
def main():
    # 1. HOME PAGE
    if st.session_state.step == "Home":
        st.markdown("<h1 style='text-align: center;'>Student Financial Behavior & Risk Profiler</h1>", unsafe_allow_html=True)
        if lottie_sphere:
            st_lottie(lottie_sphere, height=400, key="home_sphere")
        else:
            st.markdown("<h1 style='text-align: center;'>🌐</h1>", unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1,1,1])
        if col2.button("Start Financial Analysis"):
            st.session_state.step = "User_Info"
            st.rerun()

    # 2. USER REGISTRATION
    elif st.session_state.step == "User_Info":
        st.header("Step 1: Personal Details")
        with st.form("user_details"):
            name = st.text_input("Full Name")
            age = st.number_input("Age", 17, 30, 20)
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            if st.form_submit_button("Proceed to Survey"):
                st.session_state.user_name = name
                st.session_state.step = "Survey"
                st.rerun()

    # 3. THE SURVEY [cite: 13, 31, 79, 137]
    elif st.session_state.step == "Survey":
        st.header(f"Financial Journey for {st.session_state.user_name}")
        scaler, encoders, kp_model, rf_model, nn_model, safe_ref, feat_imp = load_models()

        with st.form("ml_survey"):
            col1, col2 = st.columns(2)
            
            with col1:
                place = st.selectbox("Where did you grow up?", ["🏙️ Big metro city", "🏢 Medium-sized city", "🏘️ Small town", "🌾 Rural area"])
                track = st.selectbox("How do you track expenses?", ["I check my bank balance occasionally.", "I review my history within payment apps (e.g., UPI, Paytm).", "I do not keep the track", "I use a dedicated expense-tracking app or spreadsheet."])
                graph = st.selectbox("Select your spending pattern graph:", ["Uniform Daily Expenses", "Irregular and Random Spending", "Spend a lot once and then low spending for rest", "Steady Weekdays with High Weekends"])

            with col2:
                unplanned = st.slider("Unplanned purchase frequency (1-5)", 1, 5)
                confidence = st.slider("Financial management confidence (1-5)", 1, 5)
                peer_infl = st.slider("Peer pressure influence (1-5)", 1, 5)

            st.write("---")
            st.write("**Purchase Importance (Scale)**")
            c1, c2, c3, c4 = st.columns(4)
            p_price = c1.select_slider("Price", ["Not important", "Slightly important", "Very important"])
            p_brand = c2.select_slider("Brand", ["Not important", "Slightly important", "Very important"])
            p_peer = c3.select_slider("Peer Rec", ["Not important", "Slightly important", "Very important"])
            p_util = c4.select_slider("Utility", ["Not important", "Slightly important", "Very important"])

            st.write("---")
            st.write("**Major Budget Categories**")
            b_cols = st.columns(5)
            b_food = b_cols[0].checkbox("Food")
            b_travel = b_cols[1].checkbox("Travel")
            b_fashion = b_cols[2].checkbox("Fashion")
            b_sub = b_cols[3].checkbox("Subs")
            b_fun = b_cols[4].checkbox("Entertainment")

            if st.form_submit_button("Generate My Profile"):
                # --- PREPROCESSING ---
                # 1. Encode Categoricals
                input_data = {
                    'Place_Grew_Up': place, 'Track_Expenditures': track, 'Expenditure_Graph': graph,
                    'Price_Importance': p_price, 'Brand_Importance': p_brand, 
                    'Peer_Importance': p_peer, 'Utility_Importance': p_util
                }
                
                encoded_vals = []
                for col, val in input_data.items():
                    encoded_vals.append(encoders[col].transform([val])[0])
                
                # 2. Scale Numerics
                scaled_nums = scaler.transform([[unplanned, peer_infl, confidence]])[0]
                
                # 3. Create Feature Vector for NN
                # (Matches Cleaned_Data.csv structure)
                features = np.array([encoded_vals + list(scaled_nums) + [int(b_food), int(b_travel), int(b_fashion), int(b_sub), int(b_fun)]])
                
                # --- PREDICTIONS ---
                # A. Monthly Spend (NN Softmax)
                spend_pred = np.argmax(nn_model.predict(features)) + 1
                
                # B. Cluster (K-Prototypes)
                cluster_id = kp_model.predict(features, categorical=[0,1,2,3,4,5,6])[0]
                
                # C. Risk Score (RF Weighted Distance)
                weighted_feat = features[0][:len(feat_imp)] * feat_imp
                weighted_safe = safe_ref[:len(feat_imp)] * feat_imp
                dist = cdist([weighted_feat], [weighted_safe], metric='euclidean')[0][0]
                risk_score = min(100, max(0, dist * 20)) # Normalized 1-100

                # --- DASHBOARD ---
                st.session_state.results = {
                    'spend': spend_pred, 'cluster': cluster_id, 'risk': risk_score
                }
                st.session_state.step = "Dashboard"
                st.rerun()

    # 4. RESULTS DASHBOARD
    elif st.session_state.step == "Dashboard":
        st.header(f"Financial Risk Analysis: {st.session_state.user_name}")
        res = st.session_state.results
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Expected Monthly Spend", f"Tier {res['spend']}")
        c2.metric("Behavior Cluster", f"Group {res['cluster']}")
        
        risk_color = "normal" if res['risk'] < 40 else "inverse"
        c3.metric("Financial Risk Score", f"{res['risk']:.1f}/100", delta_color=risk_color)

        if res['risk'] > 60:
            st.error("⚠️ HIGH RISK DETECTED: Consider reviewing your unplanned purchases and subscription costs.")
        elif res['risk'] > 30:
            st.warning("⚠️ MODERATE RISK: You are managing well, but peer influence is affecting your savings.")
        else:
            st.success("✅ LOW RISK: Excellent financial discipline!")

        if st.button("Restart Analysis"):
            st.session_state.step = "Home"
            st.rerun()

if __name__ == "__main__":
    main()
