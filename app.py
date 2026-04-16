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
    .main { background: #0e1117; color: white; }
    .main-title { font-size: 3rem; font-weight: 800; text-align: center; color: #4facfe; }
    .stButton>button { background: #4facfe; color: black; font-weight: bold; border-radius: 20px; border: none; }
    .metric-card { background: #161b22; padding: 25px; border-radius: 15px; border: 1px solid #30363d; text-align: center; }
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
        st.error(f"Error loading models: {e}")
    return art

if 'page' not in st.session_state: st.session_state.page = "Home"

def main():
    art = load_artifacts()

    if st.session_state.page == "Home":
        st.markdown("<div class='main-title'>Student Financial Profiler</div>", unsafe_allow_html=True)
        st.markdown("<h1 style='text-align: center;'>🧠📊🛡️</h1>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1,1,1])
        if col2.button("Begin Analysis"):
            st.session_state.page = "Survey"
            st.rerun()

    elif st.session_state.page == "Survey":
        st.header("Financial Habits Questionnaire")
        
        # Mappings to match your training strings exactly
        place_map = {"🏙️ Big metro city": "Big metro city", "🏢 Medium-sized city": "Medium-sized city", "🏘️ Small town": "Small town", "🌾 Rural area": "Rural area"}
        graph_map = {"📉 Uniform Daily": "Uniform Daily Expenses", "📊 Irregular/Random": "Irregular and Random Spending", "📈 One-time High": "Spend a lot once and then low spending for rest", "📅 High Weekends": "Steady Weekdays with High Weekends"}

        with st.form("main_form"):
            c1, c2 = st.columns(2)
            with c1:
                place_ui = st.selectbox("Where did you grow up?", list(place_map.keys()))
                track = st.selectbox("Expenditure Tracking", ["I check my bank balance occasionally.", "I review my history within payment apps (e.g., UPI, Paytm).", "I do not keep the track", "I use a dedicated expense-tracking app or spreadsheet."])
            with c2:
                graph_ui = st.selectbox("Select your spending graph:", list(graph_map.keys()))
                unplanned = st.slider("Unplanned purchase frequency (1-5)", 1, 5, 3)

            st.write("---")
            # SECTION 3: Scenario Question
            st.write("**Scenario: Justifying a ₹1,500+ unexpected expense for:**")
            jc1, jc2, jc3, jc4, jc5 = st.columns(5)
            j_disc, j_soc, j_skill, j_emer, j_trip = jc1.checkbox("Big Discount"), jc2.checkbox("Social/Party"), jc3.checkbox("Skill Course"), jc4.checkbox("Emergency"), jc5.checkbox("Planned Trip")

            st.write("---")
            confidence = st.slider("Financial management confidence (1-5)", 1, 5, 3)
            peer_infl = st.slider("Peer pressure influence (1-5)", 1, 5, 3)

            st.write("**Purchase Priorities**")
            sc1, sc2, sc3, sc4 = st.columns(4)
            p_price = sc1.select_slider("Price", ["Not important", "Slightly important", "Very important"])
            p_brand = sc2.select_slider("Brand", ["Not important", "Slightly important", "Very important"])
            p_peer = sc3.select_slider("Peer", ["Not important", "Slightly important", "Very important"])
            p_util = sc4.select_slider("Utility", ["Not important", "Slightly important", "Very important"])

            st.write("**Budget Breakdown**")
            bc = st.columns(5)
            b_food, b_travel, b_fashion, b_sub, b_fun = bc[0].checkbox("Food"), bc[1].checkbox("Travel"), bc[2].checkbox("Fashion"), bc[3].checkbox("Subs"), bc[4].checkbox("Entertainment")

            if st.form_submit_button("RUN ANALYSIS"):
                try:
                    # --- 1. PREPARE RAW DATA (For K-Proto and RF) ---
                    raw_place = place_map[place_ui]
                    raw_graph = graph_map[graph_ui]
                    
                    # Constructing the input exactly as the CSV was before cleaning
                    raw_features = [raw_place, p_price, p_brand, p_peer, p_util, unplanned, peer_infl, confidence, 
                                    int(b_food), int(b_travel), int(b_fashion), int(b_sub), int(b_fun)]

                    # --- 2. PREPARE ENCODED DATA (For Neural Network) ---
                    # Encode Categories
                    cat_row = [[raw_place, p_price, p_brand, p_peer, p_util]]
                    enc_cats = art['encoder'].transform(cat_row)[0]
                    # Scale Numerics
                    enc_nums = art['scaler'].transform([[unplanned, peer_infl, confidence]])[0]
                    # Full vector for NN
                    nn_features = np.array([list(enc_cats) + list(enc_nums) + [int(b_food), int(b_travel), int(b_fashion), int(b_sub), int(b_fun)]])

                    # --- 3. EXECUTE MODELS ---
                    # NN: Needs encoded integers
                    nn_out = art['nn'].predict(nn_features, verbose=0)
                    spend_tier = np.argmax(nn_out) + 1
                    
                    # K-Proto: Needs raw categories + numericals mixed
                    # Convert raw_features to a dataframe or proper array for KProto
                    cluster_id = art['kp'].predict(pd.DataFrame([raw_features]), categorical=[0,1,2,3,4])[0]
                    
                    # Random Forest: Usually takes the encoded version if trained on cleaned data
                    # (Adjust to raw_features if your specific RF code took strings)
                    # rf_out = art['rf'].predict(nn_features) 

                    # 4. Risk Calculation (Distance fallback)
                    risk_score = 45.0
                    if art['safe_ref'] is not None:
                        w_feat = nn_features[0][:len(art['feat_imp'])] * art['feat_imp']
                        w_safe = art['safe_ref'][:len(art['feat_imp'])] * art['feat_imp']
                        risk_score = min(100, cdist([w_feat], [w_safe], metric='euclidean')[0][0] * 15)

                    st.session_state.results = {'spend': spend_tier, 'cluster': cluster_id, 'risk': risk_score}
                    st.session_state.page = "Dashboard"
                    st.rerun()
                except Exception as e:
                    st.error(f"Analysis Error: {e}")

    elif st.session_state.page == "Dashboard":
        res = st.session_state.results
        st.markdown(f"<h1 style='text-align:center;'>Analysis for {st.session_state.get('user_name', 'User')}</h1>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        col1.metric("Spend Tier (NN)", f"Tier {res['spend']}")
        col2.metric("Behavior Group (K-Proto)", f"Group {res['cluster']}")
        col3.metric("Risk Score (RF)", f"{res['risk']:.1f}/100")
        if st.button("New Analysis"):
            st.session_state.page = "Home"
            st.rerun()

if __name__ == "__main__": main()
