import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from scipy.spatial.distance import cdist
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Student Financial Profiler", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .main { background: #0e1117; color: #ffffff; }
    .main-title { font-size: 3.5rem; font-weight: 800; background: -webkit-linear-gradient(#4facfe, #00f2fe); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center; }
    .stButton>button { background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%); color: #000; font-weight: bold; border-radius: 50px; }
    .metric-card { background: #161b22; padding: 25px; border-radius: 15px; border: 1px solid #30363d; text-align: center; }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD MODELS ---
@st.cache_resource
def load_artifacts():
    art = {}
    try:
        art['scaler'] = joblib.load('models/scaler.joblib')
        art['encoder'] = joblib.load('models/ordinal_encoder.joblib')
        art['kp'] = joblib.load('models/kproto_model.joblib')
        art['rf'] = joblib.load('models/risk_rf_model.joblib')
        art['nn'] = tf.keras.models.load_model('models/spend_nn_model.h5')
        art['feat_imp'] = joblib.load('models/feature_importances.joblib') if os.path.exists('models/feature_importances.joblib') else None
        art['safe_ref'] = joblib.load('models/safe_reference.joblib') if os.path.exists('models/safe_reference.joblib') else None
    except Exception as e:
        st.error(f"Error loading models: {e}")
    return art

if 'page' not in st.session_state: st.session_state.page = "Home"

def main():
    if st.session_state.page == "Home":
        st.markdown("<div class='main-title'>Student Financial Risk Analysis</div>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; font-size: 1.2rem; color: #8b949e;'>Using Behavioral Clustering & Neural Networks</p>", unsafe_allow_html=True)
        st.markdown("<h1 style='text-align: center;'>🧠📊🛡️</h1>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1,1,1])
        if col2.button("Begin Analysis"):
            st.session_state.page = "Registration"
            st.rerun()

    elif st.session_state.page == "Registration":
        st.header("👤 User Registration")
        with st.form("reg"):
            name = st.text_input("Full Name")
            age = st.number_input("Age", 17, 30, 20)
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            if st.form_submit_button("Proceed"):
                st.session_state.user_name = name
                st.session_state.page = "Survey"
                st.rerun()

    elif st.session_state.page == "Survey":
        st.header(f"Comprehensive Survey: {st.session_state.user_name}")
        art = load_artifacts()

        with st.form("full_survey"):
            # SECTION 1: Geography & Habits [cite: 13, 79, 137]
            c1, c2 = st.columns(2)
            place = c1.selectbox("Upbringing Environment", ["🏙️ Big metro city", "🏢 Medium-sized city", "🏘️ Small town", "🌾 Rural area"])
            track = c2.selectbox("Expenditure Tracking", ["I check my bank balance occasionally.", "I review my history within payment apps (e.g., UPI, Paytm).", "I do not keep the track", "I use a dedicated expense-tracking app or spreadsheet."])
            graph = st.selectbox("Spending Pattern Graph", ["Uniform Daily Expenses", "Irregular and Random Spending", "Spend a lot once and then low spending for rest", "Steady Weekdays with High Weekends"])

            # SECTION 2: Scaling Factors [cite: 20, 66, 90]
            unplanned = st.slider("Unplanned Purchases (1-5)", 1, 5)
            confidence = st.slider("Financial Confidence (1-5)", 1, 5)
            peer_infl = st.slider("Peer Influence (1-5)", 1, 5)

            # SECTION 3: Importance Factors [cite: 31-40]
            st.write("**Purchase Priorities**")
            ic1, ic2, ic3, ic4 = st.columns(4)
            p_price = ic1.select_slider("Price", ["Not important", "Slightly important", "Very important"])
            p_brand = ic2.select_slider("Brand", ["Not important", "Slightly important", "Very important"])
            p_peer = ic3.select_slider("Peer Rec", ["Not important", "Slightly important", "Very important"])
            p_util = ic4.select_slider("Utility", ["Not important", "Slightly important", "Very important"])

            # SECTION 4: Budget Categories [cite: 98-113]
            st.write("**Budget Allocation**")
            bc = st.columns(5)
            b_food = bc[0].checkbox("Food")
            b_travel = bc[1].checkbox("Travel")
            b_fashion = bc[2].checkbox("Fashion")
            b_sub = bc[3].checkbox("Subscriptions")
            b_fun = bc[4].checkbox("Entertainment")

            # SECTION 5: Justification Scenarios (THE MISSING DATA) 
            st.write("**Would you justify a ₹1,500+ expense for:**")
            jc1, jc2, jc3 = st.columns(3)
            j_discounts = jc1.checkbox("50% Brand Discount")
            j_party = jc2.checkbox("Social/Parties")
            j_skill = jc3.checkbox("Skill Development")
            j_emerg = jc1.checkbox("Emergencies (Repair)")
            j_trip = jc2.checkbox("Planned Trip")

            if st.form_submit_button("Run Analysis"):
                # --- PREPROCESSING (MUST MATCH CLEANED_DATA.CSV) ---
                # 1. Categorical Encode
                cat_data = [[place, p_price, p_brand, p_peer, p_util]]
                enc_cats = art['encoder'].transform(cat_data)[0]
                
                # 2. Scaling
                scaled_nums = art['scaler'].transform([[unplanned, peer_infl, confidence]])[0]
                
                # 3. CONSTRUCT FULL FEATURE VECTOR
                # Order: [Encoded Place, Price, Brand, Peer, Util] + [Scaled Unplanned, Peer_Inf, Conf] + [JUE Flags] + [Budget Flags]
                # Ensure this matches the EXACT column count of your training Data
                features = np.array([
                    list(enc_cats) + 
                    list(scaled_nums) + 
                    [int(j_emerg), int(j_discounts), int(j_party), int(j_skill), int(j_trip)] + 
                    [int(b_food), int(b_travel), int(b_fashion), int(b_sub), int(b_fun)]
                ])

                # 4. PREDICTIONS
                spend_pred = np.argmax(art['nn'].predict(features, verbose=0)) + 1
                cluster_id = art['kp'].predict(features, categorical=[0,1,2,3,4])[0]
                
                # Risk Logic
                risk_score = 50.0 # Default
                if art['safe_ref'] is not None and art['feat_imp'] is not None:
                    w_feat = features[0][:len(art['feat_imp'])] * art['feat_imp']
                    w_safe = art['safe_ref'][:len(art['feat_imp'])] * art['feat_imp']
                    risk_score = min(100, cdist([w_feat], [w_safe], metric='euclidean')[0][0] * 15)

                st.session_state.results = {'spend': spend_pred, 'cluster': cluster_id, 'risk': risk_score}
                st.session_state.page = "Dashboard"
                st.rerun()

    elif st.session_state.page == "Dashboard":
        st.markdown(f"<h1 style='text-align: center;'>Intelligence Report</h1>", unsafe_allow_html=True)
        res = st.session_state.results
        c1, c2, c3 = st.columns(3)
        c1.markdown(f"<div class='metric-card'><h3>Monthly Spend</h3><h2 style='color:#4facfe'>Tier {res['spend']}</h2></div>", unsafe_allow_html=True)
        c2.markdown(f"<div class='metric-card'><h3>Behavior Group</h3><h2 style='color:#00f2fe'>Group {res['cluster']}</h2></div>", unsafe_allow_html=True)
        c3.markdown(f"<div class='metric-card'><h3>Risk Score</h3><h2 style='color:#ff4b4b'>{res['risk']:.1f}/100</h2></div>", unsafe_allow_html=True)
        if st.button("New Analysis"):
            st.session_state.page = "Home"
            st.rerun()

if __name__ == "__main__": main()
