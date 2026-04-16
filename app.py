import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from scipy.spatial.distance import cdist
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Financial Intelligence", layout="wide", initial_sidebar_state="collapsed")

# --- HIGH-END CSS DESIGN ---
st.markdown("""
    <style>
    /* Main Background */
    .stApp {
        background: radial-gradient(circle at top right, #1e293b, #0f172a);
        color: #f8fafc;
    }
    
    /* Hero Section */
    .hero-container {
        text-align: center;
        padding: 80px 20px;
    }
    
    .glow-text {
        font-size: 4.5rem;
        font-weight: 800;
        background: linear-gradient(to right, #38bdf8, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0 0 30px rgba(56, 189, 248, 0.3);
        margin-bottom: 20px;
    }
    
    /* Custom Glassmorphism Cards */
    .card-container {
        display: flex;
        justify-content: center;
        gap: 20px;
        margin-top: 40px;
    }
    
    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 30px;
        border-radius: 20px;
        width: 300px;
        text-align: center;
        transition: 0.3s ease;
    }
    
    .glass-card:hover {
        border-color: #38bdf8;
        transform: translateY(-10px);
        background: rgba(255, 255, 255, 0.05);
    }
    
    /* Glowing Button Styling */
    .stButton>button {
        background: linear-gradient(90deg, #38bdf8, #818cf8);
        color: white;
        font-weight: 700;
        padding: 15px 40px;
        border-radius: 50px;
        border: none;
        font-size: 1.2rem;
        box-shadow: 0 10px 20px rgba(56, 189, 248, 0.4);
        transition: 0.3s;
    }
    
    .stButton>button:hover {
        box-shadow: 0 15px 30px rgba(56, 189, 248, 0.6);
        transform: scale(1.05);
    }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD MODELS SAFELY ---
@st.cache_resource
def load_artifacts():
    art = {}
    try:
        art['scaler'] = joblib.load('models/scaler.joblib')
        
        # FIX: Robust Encoder loading to prevent "Unknown Category" errors
        encoder = joblib.load('models/ordinal_encoder.joblib')
        encoder.handle_unknown = 'use_encoded_value'
        encoder.unknown_value = -1
        art['encoder'] = encoder
        
        art['kp'] = joblib.load('models/kproto_model.joblib')
        art['rf'] = joblib.load('models/risk_rf_model.joblib')
        art['nn'] = tf.keras.models.load_model('models/spend_nn_model.h5')
        
        art['feat_imp'] = joblib.load('models/feature_importances.joblib') if os.path.exists('models/feature_importances.joblib') else None
        art['safe_ref'] = joblib.load('models/safe_reference.joblib') if os.path.exists('models/safe_reference.joblib') else None
    except Exception as e:
        st.error(f"System Offline: {e}")
    return art

if 'page' not in st.session_state: st.session_state.page = "Home"

def main():
    art = load_artifacts()

    # --- LANDING PAGE ---
    if st.session_state.page == "Home":
        st.markdown("""
            <div class="hero-container">
                <div class="glow-text">Future-Ready Finance</div>
                <p style="font-size: 1.5rem; color: #94a3b8; max-width: 800px; margin: 0 auto;">
                    Harnessing Advanced Machine Learning to Predict Student Spending Behavior and Risk Profiles.
                </p>
            </div>
            """, unsafe_allow_html=True)
            
        # Feature Cards
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="glass-card"><h3>🧠 Neural Insights</h3><p>Softmax-driven Tier prediction for precise monthly spend estimates.</p></div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="glass-card"><h3>📊 Cluster DNA</h3><p>Grouping behavioral profiles using high-dimensional K-Prototypes.</p></div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="glass-card"><h3>🛡️ Risk Guardian</h3><p>weighted distance analysis to identify high-risk financial habits.</p></div>', unsafe_allow_html=True)

        st.markdown("<br><br>", unsafe_allow_html=True)
        
        c1, c2, c3 = st.columns([1,1.5,1])
        with c2:
            if st.button("INITIATE ANALYTICS", use_container_width=True):
                st.session_state.page = "Survey"
                st.rerun()

    # --- SURVEY PAGE ---
    elif st.session_state.page == "Survey":
        st.markdown("<h2 style='text-align: center; color: #38bdf8;'>Behavioral Assessment</h2>", unsafe_allow_html=True)
        
        # Mapping dictionaries to ensure clean input to encoder
        place_map = {"Big metro city": "Big metro city", "Medium-sized city": "Medium-sized city", "Small town": "Small town", "Rural area": "Rural area"}
        graph_map = {"Uniform Daily": "Uniform Daily Expenses", "Irregular/Random": "Irregular and Random Spending", "One-time High": "Spend a lot once and then low spending for rest", "High Weekends": "Steady Weekdays with High Weekends"}

        with st.form("assessment_form"):
            c1, c2 = st.columns(2)
            with c1:
                place = st.selectbox("Where did you grow up?", list(place_map.keys()))
                track = st.selectbox("Tracking Habit", ["I check my bank balance occasionally.", "I review my history within payment apps (e.g., UPI, Paytm).", "I do not keep the track", "I use a dedicated expense-tracking app or spreadsheet."])
            with c2:
                graph = st.selectbox("Spending Graph", list(graph_map.keys()))
                unplanned = st.slider("Unplanned Purchase Frequency (1-5)", 1, 5, 3)

            st.write("---")
            st.write("**Scenario: Justifying an unexpected ₹1,500+ expense for:**")
            jc1, jc2, jc3, jc4, jc5 = st.columns(5)
            j_scens = [jc1.checkbox("Big Discount"), jc2.checkbox("Social/Party"), jc3.checkbox("Skill/Course"), jc4.checkbox("Emergency"), jc5.checkbox("Planned Trip")]

            st.write("---")
            confidence = st.slider("Financial Confidence (1-5)", 1, 5, 3)
            peer_infl = st.slider("Peer Influence (1-5)", 1, 5, 3)

            st.write("**Priorities (1: Low, 3: High)**")
            sc1, sc2, sc3, sc4 = st.columns(4)
            p_price = sc1.select_slider("Price", ["Not important", "Slightly important", "Very important"])
            p_brand = sc2.select_slider("Brand", ["Not important", "Slightly important", "Very important"])
            p_peer = sc3.select_slider("Peer", ["Not important", "Slightly important", "Very important"])
            p_util = sc4.select_slider("Utility", ["Not important", "Slightly important", "Very important"])

            st.write("**Budget Allocation**")
            bc = st.columns(5)
            b_items = [bc[i].checkbox(label) for i, label in enumerate(["Food", "Travel", "Fashion", "Subs", "Fun"])]

            if st.form_submit_button("GENERATE PROFILE"):
                try:
                    # 1. RAW DATA (For Clustering/RF)
                    raw_features = [place, p_price, p_brand, p_peer, p_util, unplanned, peer_infl, confidence] + [int(x) for x in b_items]

                    # 2. ENCODED DATA (For NN)
                    cat_vals = art['encoder'].transform([[place, p_price, p_brand, p_peer, p_util]])[0]
                    num_vals = art['scaler'].transform([[unplanned, peer_infl, confidence]])[0]
                    nn_features = np.array([list(cat_vals) + list(num_vals) + [int(x) for x in b_items]])

                    # 3. PREDICTIONS
                    spend_tier = np.argmax(art['nn'].predict(nn_features, verbose=0)) + 1
                    cluster_id = art['kp'].predict(pd.DataFrame([raw_features]), categorical=[0,1,2,3,4])[0]
                    
                    # Risk Logic
                    risk = 45.0
                    if art['safe_ref'] is not None:
                        w_feat = nn_features[0][:len(art['feat_imp'])] * art['feat_imp']
                        w_safe = art['safe_ref'][:len(art['feat_imp'])] * art['feat_imp']
                        risk = min(100, cdist([w_feat], [w_safe], metric='euclidean')[0][0] * 15)

                    st.session_state.results = {'spend': spend_tier, 'cluster': cluster_id, 'risk': risk}
                    st.session_state.page = "Dashboard"
                    st.rerun()
                except Exception as e:
                    st.error(f"Processing Error: {e}")

    elif st.session_state.page == "Dashboard":
        res = st.session_state.results
        st.markdown("<h2 style='text-align: center; color: #38bdf8;'>Intelligence Report</h2>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        col1.metric("Spend Tier", f"Tier {res['spend']}")
        col2.metric("Behavior Group", f"Cluster {res['cluster']}")
        col3.metric("Risk Score", f"{res['risk']:.1f}/100")
        if st.button("New Assessment"):
            st.session_state.page = "Home"
            st.rerun()

if __name__ == "__main__": main()
