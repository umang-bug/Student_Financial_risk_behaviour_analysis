import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from scipy.spatial.distance import cdist
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Financial Risk Profiler", layout="wide")

# --- CSS STYLING ---
st.markdown("""
    <style>
    .main { background: #0e1117; color: white; }
    .stButton>button { background: #4facfe; color: black; font-weight: bold; border-radius: 20px; width: 100%; }
    .metric-card { background: #161b22; padding: 20px; border-radius: 10px; border: 1px solid #30363d; text-align: center; }
    </style>
    """, unsafe_allow_html=True)

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
        st.error(f"Model Load Error: {e}")
    return art

if 'page' not in st.session_state: st.session_state.page = "Home"

def main():
    art = load_artifacts()

    if st.session_state.page == "Home":
        st.markdown("<h1 style='text-align: center; color: #4facfe;'>Financial Intelligence System</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;'>Predicting Behavior & Risk with Machine Learning</p>", unsafe_allow_html=True)
        if st.button("Start Assessment"):
            st.session_state.page = "Survey"
            st.rerun()

    elif st.session_state.page == "Survey":
        st.header("Financial Habits Questionnaire")
        
        with st.form("main_survey"):
            c1, c2 = st.columns(2)
            with c1:
                # IMPORTANT: These strings MUST match your CSV labels exactly (No Emojis if not in CSV)
                place = st.selectbox("Where did you grow up?", ["Big metro city", "Medium-sized city", "Small town", "Rural area"])
                track = st.selectbox("How do you track expenditures?", ["I check my bank balance occasionally.", "I review my history within payment apps (e.g., UPI, Paytm).", "I do not keep the track", "I use a dedicated expense-tracking app or spreadsheet."])
                graph = st.selectbox("Spending Pattern Graph", ["Uniform Daily Expenses", "Irregular and Random Spending", "Spend a lot once and then low spending for rest", "Steady Weekdays with High Weekends"])
            
            with c2:
                unplanned = st.slider("Unplanned purchase frequency (1-5)", 1, 5, 3)
                confidence = st.slider("Financial management confidence (1-5)", 1, 5, 3)
                peer_infl = st.slider("Peer pressure influence (1-5)", 1, 5, 3)

            st.write("---")
            st.write("**Factor Importance**")
            sc1, sc2, sc3, sc4 = st.columns(4)
            p_price = sc1.select_slider("Price", ["Not important", "Slightly important", "Very important"])
            p_brand = sc2.select_slider("Brand", ["Not important", "Slightly important", "Very important"])
            p_peer = sc3.select_slider("Peer", ["Not important", "Slightly important", "Very important"])
            p_util = sc4.select_slider("Utility", ["Not important", "Slightly important", "Very important"])

            st.write("---")
            # ADDING THE MISSING QUESTION (Justification Scenarios)
            st.write("**In which scenarios would you justify an unexpected expense of ₹1,500?**")
            jc1, jc2, jc3, jc4, jc5 = st.columns(5)
            j_discount = jc1.checkbox("Big Discount")
            j_social = jc2.checkbox("Social Event")
            j_skill = jc3.checkbox("Skill/Course")
            j_emergency = jc4.checkbox("Emergency")
            j_trip = jc5.checkbox("Planned Trip")

            st.write("---")
            st.write("**Budget Breakdown**")
            bc1, bc2, bc3, bc4, bc5 = st.columns(5)
            b_food = bc1.checkbox("Food")
            b_travel = bc2.checkbox("Travel")
            b_fashion = bc3.checkbox("Fashion")
            b_sub = bc4.checkbox("Subscriptions")
            b_ent = bc5.checkbox("Entertainment")

            if st.form_submit_button("Run Analysis"):
                try:
                    # 1. Encoding (Ensure labels match OrdinalEncoder categories exactly)
                    cat_row = [[place, p_price, p_brand, p_peer, p_util]]
                    enc_cats = art['encoder'].transform(cat_row)[0]
                    
                    # 2. Scaling
                    enc_nums = art['scaler'].transform([[unplanned, peer_infl, confidence]])[0]
                    
                    # 3. Vector Construction
                    # If model expects 13 features (Cats + Nums + Budget)
                    features = np.array([list(enc_cats) + list(enc_nums) + [
                        int(b_food), int(b_travel), int(b_fashion), int(b_sub), int(b_ent)
                    ]])

                    # 4. Predict
                    nn_out = art['nn'].predict(features, verbose=0)
                    st.session_state.results = {
                        'spend': np.argmax(nn_out) + 1,
                        'cluster': art['kp'].predict(features, categorical=[0,1,2,3,4])[0],
                        'risk': 50.0 # Default
                    }
                    
                    # Risk Logic
                    if art['safe_ref'] is not None:
                        w_feat = features[0][:len(art['feat_imp'])] * art['feat_imp']
                        w_safe = art['safe_ref'][:len(art['feat_imp'])] * art['feat_imp']
                        st.session_state.results['risk'] = min(100, cdist([w_feat], [w_safe], metric='euclidean')[0][0] * 15)

                    st.session_state.page = "Dashboard"
                    st.rerun()
                except Exception as e:
                    st.error(f"Analysis Error: {e}. Check if model expects 13 or 18 features.")

    elif st.session_state.page == "Dashboard":
        res = st.session_state.results
        st.markdown("<h2 style='text-align:center;'>Analysis Results</h2>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        col1.metric("Spend Tier", f"Tier {res['spend']}")
        col2.metric("Cluster", f"Group {res['cluster']}")
        col3.metric("Risk Score", f"{res['risk']:.1f}/100")
        if st.button("Restart"):
            st.session_state.page = "Home"
            st.rerun()

if __name__ == "__main__": main()
