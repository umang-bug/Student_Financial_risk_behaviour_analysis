import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from scipy.spatial.distance import cdist
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Financial Risk Profiler", layout="wide")

# --- CSS ---
st.markdown("""
    <style>
    .main { background: #0e1117; color: #ffffff; }
    .main-title { font-size: 3rem; font-weight: 800; text-align: center; color: #4facfe; }
    .stButton>button { background: #4facfe; color: black; font-weight: bold; border-radius: 20px; }
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
    if st.session_state.page == "Home":
        st.markdown("<div class='main-title'>Financial Risk Profiler</div>", unsafe_allow_html=True)
        st.markdown("<h1 style='text-align: center;'>🧠📊</h1>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1,1,1])
        if col2.button("Start Analysis"):
            st.session_state.page = "Survey"
            st.rerun()

    elif st.session_state.page == "Survey":
        st.header("Financial Behavior Survey")
        art = load_artifacts()

        with st.form("survey_form"):
            c1, c2 = st.columns(2)
            with c1:
                # Adding 'Select...' as a default to check for empty inputs
                place = st.selectbox("Where did you grow up?", ["Select...", "🏙️ Big metro city", "🏢 Medium-sized city", "🏘️ Small town", "🌾 Rural area"])
                track = st.selectbox("Expenditure Tracking", ["Select...", "I check my bank balance occasionally.", "I review my history within payment apps (e.g., UPI, Paytm).", "I do not keep the track", "I use a dedicated expense-tracking app or spreadsheet."])
            with c2:
                graph = st.selectbox("Spending Graph", ["Select...", "Uniform Daily Expenses", "Irregular and Random Spending", "Spend a lot once and then low spending for rest", "Steady Weekdays with High Weekends"])
                unplanned = st.slider("Unplanned Purchases (1-5)", 1, 5, 1)

            confidence = st.slider("Financial Confidence (1-5)", 1, 5, 3)
            peer_infl = st.slider("Peer Influence (1-5)", 1, 5, 1)

            st.write("---")
            st.write("**Factor Importance**")
            sc1, sc2, sc3, sc4 = st.columns(4)
            p_price = sc1.select_slider("Price", ["Not important", "Slightly important", "Very important"])
            p_brand = sc2.select_slider("Brand", ["Not important", "Slightly important", "Very important"])
            p_peer = sc3.select_slider("Peer", ["Not important", "Slightly important", "Very important"])
            p_util = sc4.select_slider("Utility", ["Not important", "Slightly important", "Very important"])

            st.write("---")
            st.write("**Budget Breakdown (Empty = No spend in category)**")
            bc = st.columns(5)
            b_food = bc[0].checkbox("Food")
            b_travel = bc[1].checkbox("Travel")
            b_fashion = bc[2].checkbox("Fashion")
            b_sub = bc[3].checkbox("Subs")
            b_fun = bc[4].checkbox("Entertainment")

            submit = st.form_submit_button("RUN ML ANALYSIS")

            if submit:
                # --- POP-UP VALIDATION ---
                if "Select..." in [place, track, graph]:
                    st.warning("🚨 Please fill in all the dropdown selections before analyzing!")
                else:
                    try:
                        # 1. Encoding
                        cat_row = [[place.replace("🏙️ ","").replace("🏢 ","").replace("🏘️ ","").replace("🌾 ",""), 
                                    p_price, p_brand, p_peer, p_util]]
                        enc_cats = art['encoder'].transform(cat_row)[0]
                        
                        # 2. Scaling
                        enc_nums = art['scaler'].transform([[unplanned, peer_infl, confidence]])[0]
                        
                        # 3. Vector Construction (13 Features)
                        # We force the checkboxes to 0 if unchecked (Handle 'Empty' as Input)
                        features = np.array([list(enc_cats) + list(enc_nums) + [
                            int(b_food), int(b_travel), int(b_fashion), int(b_sub), int(b_fun)
                        ]])

                        # 4. Neural Network Prediction
                        nn_out = art['nn'].predict(features, verbose=0)
                        spend_pred = np.argmax(nn_out) + 1
                        
                        # 5. Cluster
                        cluster_id = art['kp'].predict(features, categorical=[0,1,2,3,4])[0]
                        
                        # 6. Risk Score
                        risk = 50.0
                        if art['safe_ref'] is not None:
                            w_feat = features[0][:len(art['feat_imp'])] * art['feat_imp']
                            w_safe = art['safe_ref'][:len(art['feat_imp'])] * art['feat_imp']
                            risk = min(100, cdist([w_feat], [w_safe], metric='euclidean')[0][0] * 15)

                        st.session_state.results = {'spend': spend_pred, 'cluster': cluster_id, 'risk': risk}
                        st.session_state.page = "Dashboard"
                        st.rerun()
                    except Exception as e:
                        st.error(f"Analysis Error: {e}. Check if model expects 13 features.")

    elif st.session_state.page == "Dashboard":
        res = st.session_state.results
        st.markdown("<h2 style='text-align:center;'>Financial Profile</h2>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Spend Tier", f"Tier {res['spend']}")
        c2.metric("Cluster", f"Group {res['cluster']}")
        c3.metric("Risk Score", f"{res['risk']:.1f}/100")
        if st.button("New Analysis"):
            st.session_state.page = "Home"
            st.rerun()

if __name__ == "__main__": main()
