import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Financial DNA Assessment", layout="wide")

# --- HIGH-VISIBILITY PRESENTATION CSS ---
st.markdown("""
    <style>
    /* White Background & Black Text */
    .stApp {
        background-color: #FFFFFF;
        color: #000000;
    }
    
    /* Large, Bold Headings */
    h1 {
        font-size: 4rem !important;
        color: #000000 !important;
        text-align: center;
        font-weight: 800;
        text-transform: uppercase;
        border-bottom: 5px solid #000;
        padding-bottom: 20px;
    }
    
    h2, h3 {
        font-size: 2.5rem !important;
        color: #000000 !important;
        font-weight: 700;
    }

    /* Extra Large Form Text */
    .stSelectbox label, .stSlider label, .stCheckbox label, .stRadio label {
        font-size: 1.8rem !important;
        color: #000000 !important;
        font-weight: 600 !important;
    }

    /* Large Radio/Select Inputs */
    div[data-baseweb="select"] > div {
        font-size: 1.5rem !important;
    }

    /* Bright Action Button */
    .stButton>button {
        background-color: #00FF00 !important; /* Bright Green */
        color: #000000 !important;
        font-size: 2rem !important;
        font-weight: 900 !important;
        height: 4em !important;
        width: 100% !important;
        border: 4px solid #000 !important;
        border-radius: 15px;
        margin-top: 50px;
    }

    /* Dashboard Metrics */
    [data-testid="stMetricValue"] {
        font-size: 5rem !important;
        font-weight: 900 !important;
        color: #0000FF !important; /* Bright Blue */
    }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD MODELS ---
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
        st.markdown("<h1>STUDENT FINANCIAL INTELLIGENCE</h1>", unsafe_allow_html=True)
        st.write("<br><br>", unsafe_allow_html=True)
        st.markdown("<h2 style='text-align: center;'>Predicting Financial DNA using Neural Networks & Clustering</h2>", unsafe_allow_html=True)
        st.write("<br><br>", unsafe_allow_html=True)
        if st.button("START ASSESSMENT NOW"):
            st.session_state.page = "Survey"
            st.rerun()

    # --- COMPREHENSIVE SURVEY PAGE ---
    elif st.session_state.page == "Survey":
        st.markdown("<h1>BEHAVIORAL SURVEY</h1>", unsafe_allow_html=True)
        
        with st.form("main_survey"):
            # Section 1: Demographics & Habits
            st.markdown("### 1. General Habits")
            place = st.selectbox("Where did you grow up?", 
                                ["🏙️ Big metro city", "🏢 Medium-sized city", "🏘️ Small town", "🌾 Rural area"])
            
            track = st.selectbox("How do you track your monthly expenditures?", 
                                ["I check my bank balance occasionally.", 
                                 "I review my history within payment apps (e.g., UPI, Paytm).", 
                                 "I do not keep the track", 
                                 "I use a dedicated expense-tracking app or spreadsheet."])
            
            st.write("---")
            
            # Section 2: Expenditure Graph (Visual representation)
            st.markdown("### 2. Expenditure Graph Patterns")
            graph = st.radio("Which graph describes your monthly spending?", 
                            ["Uniform Daily Expenses", 
                             "Irregular and Random Spending", 
                             "Spend a lot once and then low spending for rest", 
                             "Steady Weekdays with High Weekends"])
            
            st.write("---")
            
            # Section 3: Justification Scenarios (₹1500+)
            st.markdown("### 3. Scenario Justification")
            st.write("Would you justify an unexpected expense of ₹1,500 or more for:")
            j_em = st.checkbox("Emergencies (e.g., phone/laptop repair)")
            j_di = st.checkbox("A 50% discount on a brand I highly value")
            j_pa = st.checkbox("Social celebrations or parties")
            j_sk = st.checkbox("Skill development (workshops, certifications)")
            j_tr = st.checkbox("A planned trip with friends")

            st.write("---")
            
            # Section 4: Quantitative Sliders
            st.markdown("### 4. Behavioral Scales")
            unp = st.slider("Frequency of unplanned purchases (1-5)", 1, 5, 3)
            con = st.slider("Confidence in managing personal finances (1-5)", 1, 5, 3)
            pee = st.slider("Influence of peer pressure on spending (1-5)", 1, 5, 3)

            st.write("---")
            
            # Section 5: Importance Factors
            st.markdown("### 5. Purchase Priorities")
            p_pri = st.select_slider("Importance of PRICE", ["Not important", "Slightly important", "Very important"])
            p_bra = st.select_slider("Importance of BRAND", ["Not important", "Slightly important", "Very important"])
            p_rec = st.select_slider("Importance of PEER REC", ["Not important", "Slightly important", "Very important"])
            p_uti = st.select_slider("Importance of UTILITY", ["Not important", "Slightly important", "Very important"])

            st.write("---")
            
            # Section 6: Budget Allocation
            st.markdown("### 6. Budget Categories")
            b_fo = st.checkbox("Food & Dining")
            b_tr = st.checkbox("Travel")
            b_fa = st.checkbox("Fashion")
            b_su = st.checkbox("Subscriptions (Netflix/Spotify)")
            b_en = st.checkbox("Fun & Entertainment")

            if st.form_submit_button("GENERATE AI PROFILE"):
                # --- PREPROCESSING (RECONSTRUCT 28 FEATURES) ---
                # Remove emojis for the encoder to prevent "Unknown Category" errors
                clean_place = place.split(" ", 1)[1] if " " in place else place
                ord_in = art['oe'].transform([[clean_place, p_pri, p_bra, p_rec, p_uti]])[0]
                
                # One-Hot Encoding Expansion
                oh_track = [1 if track == "I check my bank balance occasionally." else 0,
                            1 if track == "I do not keep the track" else 0,
                            1 if track == "I review my history within payment apps (e.g., UPI, Paytm)." else 0,
                            1 if track == "I use a dedicated expense-tracking app or spreadsheet." else 0]
                oh_graph = [1 if graph == "Irregular and Random Spending" else 0,
                            1 if graph == "Spend a lot once and then low spending for rest" else 0,
                            1 if graph == "Steady Weekdays with High Weekends" else 0,
                            1 if graph == "Uniform Daily Expenses" else 0]
                
                num_sc = art['scaler'].transform([[unp, pee, con]])[0]
                
                # VECTOR ASSEMBLY (28 Features)
                # 5(Ord) + 8(OHE) + 5(JUE) + 1(Unp) + 1(Dummy) + 1(Peer) + 1(Conf) + 5(Budget) = 27... +1(Placeholder) = 28
                ann_features = np.array([
                    list(ord_in) + oh_track + oh_graph + 
                    [int(j_em), int(j_di), int(j_pa), int(j_sk), int(j_tr)] + 
                    [num_sc[0], 0, num_sc[1], num_sc[2]] + 
                    [int(b_fo), int(b_tr), int(b_fa), int(b_su), int(b_en)]
                ])

                # K-Proto Features
                kp_features = [[clean_place, p_pri, p_bra, p_rec, p_uti, track, graph, unp, pee, con, int(b_fo), int(b_tr), int(b_fa), int(b_su), int(b_en)]]

                # Predictions
                res_nn = art['nn'].predict(ann_features, verbose=0)
                st.session_state.results = {
                    'spend': np.argmax(res_nn) + 1,
                    'cluster': art['kp'].predict(pd.DataFrame(kp_features), categorical=[0,1,2,3,4,5,6])[0]
                }
                st.session_state.page = "Dashboard"
                st.rerun()

    # --- DASHBOARD PAGE ---
    elif st.session_state.page == "Dashboard":
        st.markdown("<h1>ANALYSIS REPORT</h1>", unsafe_allow_html=True)
        st.write("<br><br>", unsafe_allow_html=True)
        res = st.session_state.results
        c1, c2 = st.columns(2)
        c1.metric("SPEND TIER", f"TIER {res['spend']}")
        c2.metric("PERSONA GROUP", f"GROUP {res['cluster']}")
        
        st.write("<br><br>", unsafe_allow_html=True)
        if st.button("RESTART SYSTEM"):
            st.session_state.page = "Home"
            st.rerun()

if __name__ == "__main__":
    main()
