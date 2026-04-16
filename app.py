import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from scipy.spatial.distance import cdist
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="MNIT Financial Profiler", layout="wide", initial_sidebar_state="collapsed")

# --- NEON CYBERPUNK CSS ---
st.markdown("""
    <style>
    .stApp { background: #050a18; color: #ffffff; }
    
    /* Improved Home Page Styling */
    .hero-container {
        text-align: center;
        padding: 60px 20px;
        background: rgba(255, 255, 255, 0.02);
        border-radius: 30px;
        border: 1px solid #1e293b;
        margin: 20px;
    }
    
    .main-title {
        font-size: 5rem;
        font-weight: 900;
        background: linear-gradient(45deg, #00f2fe 0%, #4facfe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    
    .highlight-text {
        color: #00f2fe;
        font-size: 1.4rem;
        font-weight: 400;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 40px;
    }

    .glass-card {
        background: #111827;
        padding: 30px;
        border-radius: 20px;
        border-top: 4px solid #4facfe;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
        height: 100%;
    }
    
    .glass-card h3 { color: #00f2fe; margin-bottom: 15px; }
    .glass-card p { color: #94a3b8; font-size: 0.95rem; }

    /* Button Styling */
    .stButton>button {
        background: linear-gradient(90deg, #00f2fe 0%, #4facfe 100%);
        color: #050a18 !important;
        font-weight: 800;
        padding: 20px 60px;
        border-radius: 100px;
        border: none;
        font-size: 1.3rem;
        transition: 0.4s ease;
        display: block;
        margin: 0 auto;
    }
    
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 0 40px rgba(0, 242, 254, 0.6);
    }
    </style>
    """, unsafe_allow_html=True)

# --- LOADING ARTIFACTS ---
@st.cache_resource
def load_artifacts():
    art = {}
    p = 'models/'
    art['scaler'] = joblib.load(p + 'scaler.joblib')
    art['oe'] = joblib.load(p + 'ordinal_encoder.joblib')
    art['kp'] = joblib.load(p + 'kproto_model.joblib')
    art['nn'] = tf.keras.models.load_model(p + 'spend_nn_model.h5')
    return art

if 'page' not in st.session_state: st.session_state.page = "Home"

def main():
    art = load_artifacts()

    # --- LANDING PAGE ---
    if st.session_state.page == "Home":
        st.markdown("""
            <div class="hero-container">
                <div class="main-title">FINANCIAL DNA</div>
                <div class="highlight-text">Advanced Predictive Analytics Engine</div>
                <div style="margin-top: 40px;"></div>
            </div>
            """, unsafe_allow_html=True)
            
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="glass-card"><h3>Neural Network</h3><p>Proprietary ANN architecture analyzing 28 distinct behavioral markers for spend prediction.</p></div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="glass-card"><h3>Clustering</h3><p>K-Prototypes grouping based on psychological spending profiles and habits.</p></div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="glass-card"><h3>MNIT Jaipur</h3><p>Research-backed model tailored for academic and student financial environments.</p></div>', unsafe_allow_html=True)

        st.write("<br><br>", unsafe_allow_html=True)
        if st.button("LAUNCH ASSESSMENT"):
            st.session_state.page = "Survey"
            st.rerun()

    # --- SURVEY PAGE ---
    elif st.session_state.page == "Survey":
        st.markdown("<h1 style='text-align: center; color: #00f2fe;'>Behavioral Survey</h1>", unsafe_allow_html=True)
        
        with st.form("comprehensive_survey"):
            c1, c2 = st.columns(2)
            place = c1.selectbox("Upbringing Environment", ["🏙️ Big metro city","🏢 Medium-sized city","🏘️ Small town","🌾 Rural area"])
            track = c2.selectbox("Tracking Habit", ["I check my bank balance occasionally.", "I review my history within payment apps (e.g., UPI, Paytm).", "I do not keep the track", "I use a dedicated expense-tracking app or spreadsheet."])
            graph = c1.selectbox("Spending Pattern Graph", ["Uniform Daily Expenses", "Irregular and Random Spending", "Spend a lot once and then low spending for rest", "Steady Weekdays with High Weekends"])
            
            st.write("---")
            st.write("**Scenario: Would you justify a ₹1,500 expense for:**")
            jc = st.columns(5)
            j_em = jc[0].checkbox("Emergency")
            j_di = jc[1].checkbox("50% Discount")
            j_pa = jc[2].checkbox("Social/Party")
            j_sk = jc[3].checkbox("Workshop")
            j_tr = jc[4].checkbox("Planned Trip")

            st.write("---")
            unp = st.slider("Unplanned Purchases (1-5)", 1, 5, 3)
            con = st.slider("Finance Confidence (1-5)", 1, 5, 3)
            pee = st.slider("Peer Influence (1-5)", 1, 5, 3)

            st.write("**Priorities (1: Low, 3: High)**")
            pc = st.columns(4)
            p_pri = pc[0].select_slider("Price", ["Not important","Slightly important","Very important"])
            p_bra = pc[1].select_slider("Brand", ["Not important","Slightly important","Very important"])
            p_rec = pc[2].select_slider("Peer Rec", ["Not important","Slightly important","Very important"])
            p_uti = pc[3].select_slider("Utility", ["Not important","Slightly important","Very important"])

            st.write("**Budget Breakdown**")
            bc = st.columns(5)
            b_fo, b_tr, b_fa, b_su, b_en = bc[0].checkbox("Food"), bc[1].checkbox("Travel"), bc[2].checkbox("Fashion"), bc[3].checkbox("Subs"), bc[4].checkbox("Entertainment")

            if st.form_submit_button("PROCESS ENGINE"):
                # --- PREPROCESSING 28 FEATURES ---
                ord_in = art['oe'].transform([[place, p_pri, p_bra, p_rec, p_uti]])[0]
                
                # Manual One-Hot Expansion (8 columns)
                oh_track = [1 if track == "I check my bank balance occasionally." else 0,
                            1 if track == "I do not keep the track" else 0,
                            1 if track == "I review my history within payment apps (e.g., UPI, Paytm)." else 0,
                            1 if track == "I use a dedicated expense-tracking app or spreadsheet." else 0]
                oh_graph = [1 if graph == "Irregular and Random Spending" else 0,
                            1 if graph == "Spend a lot once and then low spending for rest" else 0,
                            1 if graph == "Steady Weekdays with High Weekends" else 0,
                            1 if graph == "Uniform Daily Expenses" else 0]
                
                num_sc = art['scaler'].transform([[unp, pee, con]])[0]
                
                # RECONSTRUCTING VECTOR (28 Features)
                # Adding 0 as placeholder for 'Monthly_Spend' (the 28th column)
                ann_features = np.array([
                    list(ord_in) + oh_track + oh_graph + 
                    [int(j_em), int(j_di), int(j_pa), int(j_sk), int(j_tr)] + 
                    [num_sc[0], 0, num_sc[1], num_sc[2]] + 
                    [int(b_fo), int(b_tr), int(b_fa), int(b_su), int(b_en)]
                ])

                # K-Proto Features
                kp_features = [[place, p_pri, p_bra, p_rec, p_uti, track, graph, unp, pee, con, int(b_fo), int(b_tr), int(b_fa), int(b_su), int(b_en)]]

                # Predictions
                res_nn = art['nn'].predict(ann_features, verbose=0)
                st.session_state.results = {
                    'spend': np.argmax(res_nn) + 1,
                    'cluster': art['kp'].predict(pd.DataFrame(kp_features), categorical=[0,1,2,3,4,5,6])[0]
                }
                st.session_state.page = "Dashboard"
                st.rerun()

    elif st.session_state.page == "Dashboard":
        res = st.session_state.results
        st.markdown("<h1 style='text-align: center; color: #00f2fe;'>Analytics Report</h1>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        c1.metric("Predicted Spend Tier", f"Tier {res['spend']}")
        c2.metric("Persona Cluster", f"Group {res['cluster']}")
        if st.button("New Analysis"):
            st.session_state.page = "Home"
            st.rerun()

if __name__ == "__main__": main()
