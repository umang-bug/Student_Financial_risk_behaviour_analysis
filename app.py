import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from scipy.spatial.distance import cdist
import os

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="Student Financial Profiler", 
    layout="wide", 
    initial_sidebar_state="collapsed"
)

# --- PROFESSIONAL CSS STYLING ---
st.markdown("""
    <style>
    .main { background: #0e1117; color: #ffffff; }
    .hero-section {
        text-align: center;
        padding: 100px 20px;
        background: linear-gradient(180deg, #161b22 0%, #0e1117 100%);
        border-radius: 20px;
        margin-bottom: 40px;
    }
    .main-title {
        font-size: 4rem;
        font-weight: 800;
        letter-spacing: -1px;
        background: -webkit-linear-gradient(#4facfe, #00f2fe);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 10px;
    }
    .sub-title {
        color: #8b949e;
        font-size: 1.5rem;
        margin-bottom: 50px;
    }
    .stButton>button {
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
        color: #000;
        font-weight: bold;
        border: none;
        padding: 15px 30px;
        font-size: 1.2rem;
        border-radius: 50px;
        transition: 0.3s;
    }
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0px 0px 20px rgba(79, 172, 254, 0.5);
    }
    .metric-card {
        background: #161b22;
        padding: 30px;
        border-radius: 15px;
        border: 1px solid #30363d;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD MODELS SAFELY ---
@st.cache_resource
def load_artifacts():
    # Syncing with your GitHub filenames
    artifacts = {}
    try:
        artifacts['scaler'] = joblib.load('models/scaler.joblib')
        artifacts['encoder'] = joblib.load('models/ordinal_encoder.joblib')
        artifacts['kp'] = joblib.load('models/kproto_model.joblib')
        artifacts['rf'] = joblib.load('models/risk_rf_model.joblib')
        artifacts['nn'] = tf.keras.models.load_model('models/spend_nn_model.h5')
        
        # Checking for the distance logic files
        if os.path.exists('models/safe_reference.joblib'):
            artifacts['safe_ref'] = joblib.load('models/safe_reference.joblib')
            artifacts['feat_imp'] = joblib.load('models/feature_importances.joblib')
        else:
            artifacts['safe_ref'] = None
    except Exception as e:
        st.error(f"Error loading models: {e}")
    return artifacts

# --- SESSION STATE ---
if 'page' not in st.session_state:
    st.session_state.page = "Home"

# --- APP PAGES ---
def main():
    if st.session_state.page == "Home":
        # HERO SECTION
        st.markdown("""
            <div class="hero-section">
                <div class="main-title">Financial Intelligence System</div>
                <div class="sub-title">Predicting Student Spending Behavior & Risk Profiles with ML</div>
            </div>
            """, unsafe_allow_html=True)
        
        # 3 COLUMN FEATURES
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("<div class='metric-card'><h3>🔍 Behavioral Clustering</h3><p>Grouping patterns using K-Prototypes</p></div>", unsafe_allow_html=True)
        with col2:
            st.markdown("<div class='metric-card'><h3>📈 Spend Prediction</h3><p>Neural Network Softmax Analysis</p></div>", unsafe_allow_html=True)
        with col3:
            st.markdown("<div class='metric-card'><h3>🚩 Risk Assessment</h3><p>Random Forest Weighted Distance</p></div>", unsafe_allow_html=True)

        st.write("<br><br>", unsafe_allow_html=True)
        
        btn_col1, btn_col2, btn_col3 = st.columns([1,1,1])
        with btn_col2:
            if st.button("Get Started →", use_container_width=True):
                st.session_state.page = "Registration"
                st.rerun()

    elif st.session_state.page == "Registration":
        st.markdown("<h2 style='text-align: center;'>Create Your Profile</h2>", unsafe_allow_html=True)
        with st.form("reg_form"):
            name = st.text_input("Full Name")
            age = st.number_input("Age", 17, 30, 20)
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
            if st.form_submit_button("Start Assessment"):
                st.session_state.user_name = name
                st.session_state.page = "Survey"
                st.rerun()

    elif st.session_state.page == "Survey":
        st.header(f"Financial Habits Survey for {st.session_state.user_name}")
        art = load_artifacts()

        with st.form("survey_form"):
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
            st.write("**Purchase Factor Importance**")
            sc1, sc2, sc3, sc4 = st.columns(4)
            p_price = sc1.select_slider("Price", ["Not important", "Slightly important", "Very important"])
            p_brand = sc2.select_slider("Brand", ["Not important", "Slightly important", "Very important"])
            p_peer = sc3.select_slider("Peer Rec", ["Not important", "Slightly important", "Very important"])
            p_util = sc4.select_slider("Utility", ["Not important", "Slightly important", "Very important"])

            st.write("---")
            st.write("**Budget Allocation**")
            b_cols = st.columns(5)
            b_food = b_cols[0].checkbox("Food")
            b_travel = b_cols[1].checkbox("Travel")
            b_fashion = b_cols[2].checkbox("Fashion")
            b_sub = b_cols[3].checkbox("Subscriptions")
            b_fun = b_cols[4].checkbox("Entertainment")

            if st.form_submit_button("Analyze My Finances"):
                # 1. ENCODING
                # Creating row to match OrdinalEncoder categories [cite: 13, 15-17, 31-40]
                cat_row = [[place, p_price, p_brand, p_peer, p_util]]
                enc_cats = art['encoder'].transform(cat_row)[0]
                
                # 2. SCALING
                enc_nums = art['scaler'].transform([[unplanned, peer_infl, confidence]])[0]
                
                # 3. FEATURE VECTOR
                # Order must match your 'Cleaned_Data.csv' [cite: 7]
                features = np.array([list(enc_cats) + list(enc_nums) + [int(b_food), int(b_travel), int(b_fashion), int(b_sub), int(b_fun)]])
                
                # 4. PREDICTIONS
                spend_pred = np.argmax(art['nn'].predict(features, verbose=0)) + 1
                cluster_id = art['kp'].predict(features, categorical=[0,1,2,3,4])[0]
                
                if art['safe_ref'] is not None:
                    # Calculation using weighted distance logic from your notebook
                    weighted_feat = features[0][:len(art['feat_imp'])] * art['feat_imp']
                    weighted_safe = art['safe_ref'][:len(art['feat_imp'])] * art['feat_imp']
                    risk_val = cdist([weighted_feat], [weighted_safe], metric='euclidean')[0][0]
                    risk_score = min(100, risk_val * 20)
                else:
                    risk_score = 45.0 # Fallback

                st.session_state.results = {'spend': spend_pred, 'cluster': cluster_id, 'risk': risk_score}
                st.session_state.page = "Dashboard"
                st.rerun()

    elif st.session_state.page == "Dashboard":
        st.markdown(f"<h1 style='text-align: center;'>Intelligence Report: {st.session_state.user_name}</h1>", unsafe_allow_html=True)
        res = st.session_state.results
        
        rc1, rc2, rc3 = st.columns(3)
        with rc1:
            st.markdown(f"<div class='metric-card'><h3>Monthly Spend</h3><h2 style='color: #4facfe;'>Tier {res['spend']}</h2></div>", unsafe_allow_html=True)
        with rc2:
            st.markdown(f"<div class='metric-card'><h3>Behavior Cluster</h3><h2 style='color: #00f2fe;'>Group {res['cluster']}</h2></div>", unsafe_allow_html=True)
        with rc3:
            st.markdown(f"<div class='metric-card'><h3>Risk Score</h3><h2 style='color: #ff4b4b;'>{res['risk']:.1f}/100</h2></div>", unsafe_allow_html=True)

        if st.button("New Analysis"):
            st.session_state.page = "Home"
            st.rerun()

if __name__ == "__main__":
    main()
