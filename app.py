import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from scipy.spatial.distance import cdist
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Financial Intelligence", layout="wide", initial_sidebar_state="collapsed")

# --- CUSTOM CSS (HIGH-END DESIGN) ---
st.markdown("""
    <style>
    .stApp { background: radial-gradient(circle at top right, #0f172a, #020617); color: #f8fafc; }
    .hero-text { font-size: 4rem; font-weight: 800; background: linear-gradient(to right, #38bdf8, #818cf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; text-align: center; }
    .glass-card { background: rgba(255, 255, 255, 0.03); border: 1px solid rgba(255, 255, 255, 0.1); padding: 25px; border-radius: 20px; transition: 0.3s; }
    .stButton>button { background: linear-gradient(90deg, #38bdf8, #818cf8); color: white; border-radius: 50px; border: none; padding: 12px 40px; font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)

# --- LOADING ARTIFACTS ---
@st.cache_resource
def load_artifacts():
    art = {}
    path = 'models/'
    art['scaler'] = joblib.load(path + 'scaler.joblib')
    art['oe'] = joblib.load(path + 'ordinal_encoder.joblib')
    art['kp'] = joblib.load(path + 'kproto_model.joblib')
    art['rf'] = joblib.load(path + 'risk_rf_model.joblib')
    art['nn'] = tf.keras.models.load_model(path + 'spend_nn_model.h5')
    return art

if 'page' not in st.session_state: st.session_state.page = "Home"

def main():
    art = load_artifacts()

    # --- LANDING PAGE ---
    if st.session_state.page == "Home":
        st.markdown("<div class='hero-text'>MNIT Financial Profiler</div>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #94a3b8; font-size: 1.2rem;'>Advanced Predictive Behavioral Analysis for Students</p>", unsafe_allow_html=True)
        
        st.write("<br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1: st.markdown('<div class="glass-card"><h3>🧠 ANN</h3><p>28-Feature Softmax Neural Network for Tiered spend prediction.</p></div>', unsafe_allow_html=True)
        with col2: st.markdown('<div class="glass-card"><h3>📊 K-Proto</h3><p>Behavioral clustering using raw categorical and numeric features.</p></div>', unsafe_allow_html=True)
        with col3: st.markdown('<div class="glass-card"><h3>🛡️ RF</h3><p>Risk scoring through ensemble classification logic.</p></div>', unsafe_allow_html=True)

        st.write("<br>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns([1,1,1])
        if c2.button("ENTER ANALYTICS ENGINE", use_container_width=True):
            st.session_state.page = "Survey"
            st.rerun()

    # --- SURVEY PAGE ---
    elif st.session_state.page == "Survey":
        st.markdown("<h2 style='text-align: center; color: #38bdf8;'>Data Acquisition</h2>", unsafe_allow_html=True)
        
        with st.form("behavior_survey"):
            c1, c2 = st.columns(2)
            # Question 1: Place
            place = c1.selectbox("Upbringing Environment", ["🏙️ Big metro city","🏢 Medium-sized city","🏘️ Small town","🌾 Rural area"])
            # Question 2: Tracking (Will be One-Hot encoded)
            track = c2.selectbox("Expenditure Tracking", ["I check my bank balance occasionally.", "I review my history within payment apps (e.g., UPI, Paytm).", "I do not keep the track", "I use a dedicated expense-tracking app or spreadsheet."])
            # Question 3: Graph (Will be One-Hot encoded)
            graph = c1.selectbox("Spending Pattern Graph", ["Uniform Daily Expenses", "Irregular and Random Spending", "Spend a lot once and then low spending for rest", "Steady Weekdays with High Weekends"])
            
            # Scenario (Justification)
            st.write("**Scenario Justification**")
            jc = st.columns(5)
            j_em = jc[0].checkbox("Emergency")
            j_di = jc[1].checkbox("50% Discount")
            j_pa = jc[2].checkbox("Social/Party")
            j_sk = jc[3].checkbox("Skill Workshop")
            j_tr = jc[4].checkbox("Planned Trip")

            # Numeric Sliders
            unp = st.slider("Unplanned Purchases (1-5)", 1, 5, 3)
            con = st.slider("Finance Confidence (1-5)", 1, 5, 3)
            pee = st.slider("Peer Influence (1-5)", 1, 5, 3)

            # Priorities
            st.write("**Purchase Priorities**")
            pc = st.columns(4)
            p_pri = pc[0].select_slider("Price", ["Not important","Slightly important","Very important"])
            p_bra = pc[1].select_slider("Brand", ["Not important","Slightly important","Very important"])
            p_rec = pc[2].select_slider("Peer Rec", ["Not important","Slightly important","Very important"])
            p_uti = pc[3].select_slider("Utility", ["Not important","Slightly important","Very important"])

            # Budget
            st.write("**Budget Categories**")
            bc = st.columns(5)
            b_fo, b_tr, b_fa, b_su, b_en = bc[0].checkbox("Food"), bc[1].checkbox("Travel"), bc[2].checkbox("Fashion"), bc[3].checkbox("Subs"), bc[4].checkbox("Entertainment")

            if st.form_submit_button("PROCESS DATA"):
                # --- STEAM 1: RECONSTRUCT 28 FEATURES FOR ANN ---
                # Ordinal Encode (First 5)
                ord_in = art['oe'].transform([[place, p_pri, p_bra, p_rec, p_uti]])[0]
                # One-Hot Manual Encoding (8 columns)
                oh_track = [1 if track == "I check my bank balance occasionally." else 0,
                            1 if track == "I do not keep the track" else 0,
                            1 if track == "I review my history within payment apps (e.g., UPI, Paytm)." else 0,
                            1 if track == "I use a dedicated expense-tracking app or spreadsheet." else 0]
                oh_graph = [1 if graph == "Irregular and Random Spending" else 0,
                            1 if graph == "Spend a lot once and then low spending for rest" else 0,
                            1 if graph == "Steady Weekdays with High Weekends" else 0,
                            1 if graph == "Uniform Daily Expenses" else 0]
                # Scale Numerics
                num_sc = art['scaler'].transform([[unp, pee, con]])[0]
                
                # ANN INPUT (28 Features)
                # Structure matches your 'Data' concat in cleaning script
                ann_features = np.array([list(ord_in) + oh_track + oh_graph + [int(j_em), int(j_di), int(j_pa), int(j_sk), int(j_tr)] + [num_sc[0], 0, num_sc[1], num_sc[2]] + [int(b_fo), int(b_tr), int(b_fa), int(b_su), int(b_en)]])
                
                # --- STREAM 2: RAW INPUT FOR K-PROTO ---
                # K-Proto expects the raw strings + numeric values as used in training
                kp_features = [[place, p_pri, p_bra, p_rec, p_uti, track, graph, unp, pee, con, int(b_fo), int(b_tr), int(b_fa), int(b_su), int(b_en)]]

                # --- EXECUTION ---
                spend_pred = np.argmax(art['nn'].predict(ann_features, verbose=0)) + 1
                cluster_id = art['kp'].predict(pd.DataFrame(kp_features), categorical=[0,1,2,3,4,5,6])[0]

                st.session_state.results = {'spend': spend_tier, 'cluster': cluster_id}
                st.session_state.page = "Dashboard"
                st.rerun()

    # --- DASHBOARD PAGE ---
    elif st.session_state.page == "Dashboard":
        res = st.session_state.results
        st.markdown("<h2 style='text-align: center; color: #38bdf8;'>Intelligence Dashboard</h2>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        col1.metric("Predicted Spend Tier", f"Tier {res['spend']}")
        col2.metric("Behavior Cluster", f"Cluster {res['cluster']}")
        if st.button("New Analysis"):
            st.session_state.page = "Home"
            st.rerun()

if __name__ == "__main__": main()
