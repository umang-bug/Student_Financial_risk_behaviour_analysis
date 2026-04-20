import streamlit as st
import numpy as np
import pandas as pd
import joblib

st.set_page_config(page_title="Student Financial Profiler | MNIT Jaipur", page_icon="💰", layout="centered")

st.markdown("""
<style>
html, body, [class*="css"] { font-size: 17px !important; }
h1 { font-size: 2.2rem !important; font-weight: 800; }
h2 { font-size: 1.6rem !important; font-weight: 700; }
h3 { font-size: 1.3rem !important; }
.stButton>button {
    background-color: #00c04b !important; color: white !important;
    font-size: 1.1rem !important; font-weight: 700 !important;
    padding: 0.6rem 2rem !important; border-radius: 8px !important; border: none !important;
}
.card { background:#f8f9fa; border-left:6px solid #00c04b; border-radius:8px; padding:1rem 1.4rem; margin:0.6rem 0; }
.card-red    { border-left-color:#e74c3c !important; background:#fdf0ef !important; }
.card-orange { border-left-color:#f39c12 !important; background:#fef9ef !important; }
.card-green  { border-left-color:#00c04b !important; background:#f0f9f4 !important; }
</style>
""", unsafe_allow_html=True)

# ── Personas ──────────────────────────────────────────────────
PERSONAS = {
    0: {"name":"Mindful Minimalist",    "emoji":"🟢",
        "desc":"Lowest impulse buying and peer influence. High financial confidence. Spends mainly on food with controlled spending bursts. Utility-driven decisions."},
    1: {"name":"Cautious Bank Checker", "emoji":"🟢",
        "desc":"Regularly checks bank balance but no structured budget. Irregular random spending. Mostly food-only budget. Financially aware but not yet disciplined."},
    2: {"name":"Balanced Explorer",     "emoji":"🟡",
        "desc":"The average student — middle of everything. Uses payment apps, weekend + irregular spender. Food & travel focused, justifies trip expenses."},
    3: {"name":"All-In Spender",        "emoji":"🟠",
        "desc":"Spends across ALL budget categories simultaneously. Tracks via apps but still high risk. Heavily discount-driven, justifies trips frequently."},
    4: {"name":"Weekend Party Shopper", "emoji":"🔴",
        "desc":"High weekend spending spikes. 100% party-driven expenses. Fashion is top budget. Very high impulse and peer influence. Discount-driven splurger."},
    5: {"name":"Financial Phantom",     "emoji":"🔴",
        "desc":"Zero financial tracking. Completely irregular unstructured spending. Entertainment-first. Highest impulse in all clusters."},
}

FEATURE_ADVICE = {
    "Unplanned_Purchases": ("decrease","Plan purchases in advance — avoid impulse buys",2),
    "Peer_Influence":      ("decrease","Make spending decisions independently of social pressure",2),
    "Finance_Confidence":  ("increase","Track expenses daily — use a budgeting app",1),
    "Price_Importance":    ("increase","Compare prices before buying — prioritize value",1),
    "Brand_Importance":    ("decrease","Choose products by utility, not brand name",2),
    "Utility_Importance":  ("increase","Ask 'do I really need this?' before every purchase",1),
    "Budget_Fashion":      ("decrease","Limit clothing purchases to essentials only",2),
    "Budget_Entertainment":("decrease","Set a fixed fun budget per month",1),
    "Budget_Subscriptions":("decrease","Audit subscriptions — cancel unused ones",1),
    "Discounts(JBS)":      ("decrease","Only buy discounted items if already planned",1),
    "Party(JBS)":          ("decrease","Set a fixed budget for social events",2),
}

# ── Load models ───────────────────────────────────────────────
@st.cache_resource
def load_models():
    m = {}
    try:
        import tensorflow as tf
        m["nn"]              = tf.keras.models.load_model("models/nn_model.h5")
        m["nn_cols"]         = joblib.load("models/nn_feature_cols.joblib")
        m["mms"]             = joblib.load("models/mms_scaler.joblib")
        m["rf_factorize"]    = joblib.load("models/rf_factorize_maps.joblib")
        m["rf_scaler"]       = joblib.load("models/rf_scaler.joblib")
        m["rf_importances"]  = joblib.load("models/rf_importances.joblib")
        m["rf_safe_ref"]     = joblib.load("models/rf_safe_reference.joblib")
        m["rf_score_scaler"] = joblib.load("models/rf_score_scaler.joblib")
        m["rf_cols"]         = joblib.load("models/rf_feature_cols.joblib")
        m["km"]              = joblib.load("models/kmeans_model.joblib")
        m["km_scaler"]       = joblib.load("models/kmeans_scaler.joblib")
        m["km_fusion"]       = joblib.load("models/kmeans_fusion_map.joblib")
        m["km_cols"]         = joblib.load("models/kmeans_feature_cols.joblib")
        m["km_profiles"]     = joblib.load("models/cluster_profiles.joblib")
    except Exception as e:
        st.error(f"Model loading error: {e}")
    return m

M = load_models()

# ── Options ───────────────────────────────────────────────────
PLACE_OPTS   = ["🏙️ Big metro city","🏢 Medium-sized city","🏘️ Small town","🌾 Rural area"]
IMP_OPTS     = ["Not important","Slightly important","Very important"]
TRACK_OPTS   = [
    "I do not keep the track",
    "I check my bank balance occasionally.",
    "I review my history within payment apps (e.g., UPI, Paytm).",
    "I use a dedicated expense-tracking app or spreadsheet.",
]
GRAPH_OPTS   = [
    "Uniform Daily Expenses",
    "Irregular and Random Spending",
    "Spend a lot once and then low spending for rest",
    "Steady Weekdays with High Weekends",
]
JUSTIFY_OPTS = [
    "Emergencies (e.g., phone/laptop repair).",
    "A 50% discount on a brand I highly value.",
    "Social celebrations or parties.",
    "Skill development (workshops, certifications, technical kits).",
    "A planned trip with friends.",
]
BUDGET_OPTS  = ["Food & Dining","Travel","Fashion","Subscriptions (Netflix, Spotify, etc.)","Fun & Entertainment"]

# ── Header ────────────────────────────────────────────────────
st.title("💰 Student Financial Behavior & Risk Profiling")
st.markdown("**MNIT Jaipur** — Fill the survey below to get your financial intelligence report.")
st.divider()
st.header("📋 Financial Behavior Survey")

with st.form("survey"):
    st.subheader("1. Background")
    place = st.radio("Where did you grow up?", PLACE_OPTS)

    st.subheader("2. Spending Behavior")
    unplanned = st.slider("How often do you make unplanned purchases?",     1, 5, 3, help="1=Never  5=Very often")
    peer_inf  = st.slider("How much does peer pressure influence spending?", 1, 5, 3, help="1=Not at all  5=Very much")
    fin_conf  = st.slider("How confident are you managing your finances?",  1, 5, 3, help="1=Not confident  5=Very confident")

    st.subheader("3. Purchase Priorities")
    price_imp   = st.select_slider("Importance of Price/Cost",          IMP_OPTS, value="Slightly important")
    brand_imp   = st.select_slider("Importance of Brand",               IMP_OPTS, value="Slightly important")
    peer_imp    = st.select_slider("Importance of Peer Recommendation", IMP_OPTS, value="Not important")
    utility_imp = st.select_slider("Importance of Long-term Utility",   IMP_OPTS, value="Very important")

    st.subheader("4. Tracking & Patterns")
    track = st.selectbox("How do you track monthly expenditures?", TRACK_OPTS)
    graph = st.radio("Which spending pattern best matches yours?", GRAPH_OPTS)

    st.subheader("5. Budget & Justifications")
    budget_sel  = st.multiselect("Where do you spend most of your budget? (select all that apply)", BUDGET_OPTS)
    justify_sel = st.multiselect("When would you justify an unexpected ₹1,500+ expense?", JUSTIFY_OPTS)

    submitted = st.form_submit_button("🔍 Generate My Financial Report")

# ══════════════════════════════════════════════════════════════
# PREDICTION
# ══════════════════════════════════════════════════════════════
if submitted and M:

    IMP_MAP   = {"Not important":0,"Slightly important":1,"Very important":2}
    PLACE_MAP = {"🏙️ Big metro city":0,"🏢 Medium-sized city":1,"🏘️ Small town":2,"🌾 Rural area":3}
    TRACK_MAP = {
        "I do not keep the track":0,
        "I check my bank balance occasionally.":1,
        "I review my history within payment apps (e.g., UPI, Paytm).":2,
        "I use a dedicated expense-tracking app or spreadsheet.":3,
    }
    GRAPH_MAP = {
        "Uniform Daily Expenses":0,
        "Irregular and Random Spending":1,
        "Spend a lot once and then low spending for rest":2,
        "Steady Weekdays with High Weekends":3,
    }

    # scaled
    scaled = M["mms"].transform([[unplanned, peer_inf, fin_conf]])[0]
    up_sc, pi_sc, fc_sc = scaled

    # budget + JBS flags
    bf = {
        "Budget_FoodDining":    1 if "Food & Dining"                           in budget_sel else 0,
        "Budget_Travel":        1 if "Travel"                                  in budget_sel else 0,
        "Budget_Fashion":       1 if "Fashion"                                 in budget_sel else 0,
        "Budget_Subscriptions": 1 if "Subscriptions (Netflix, Spotify, etc.)" in budget_sel else 0,
        "Budget_Entertainment": 1 if "Fun & Entertainment"                     in budget_sel else 0,
    }
    jbs = {
        "Emergency(JBS)": 1 if "Emergencies"      in " ".join(justify_sel) else 0,
        "Discounts(JBS)": 1 if "discount"          in " ".join(justify_sel).lower() else 0,
        "Party(JBS)":     1 if "celebrations"      in " ".join(justify_sel).lower() else 0,
        "Workshop(JBS)":  1 if "Skill development" in " ".join(justify_sel) else 0,
        "Trip(JBS)":      1 if "planned trip"       in " ".join(justify_sel).lower() else 0,
    }
    track_ohe = {
        "Track_Bank_Balance":     1 if TRACK_MAP[track]==1 else 0,
        "Track_None":             1 if TRACK_MAP[track]==0 else 0,
        "Track_Payment_Apps":     1 if TRACK_MAP[track]==2 else 0,
        "Track_Apps_Spreadsheet": 1 if TRACK_MAP[track]==3 else 0,
    }
    graph_ohe = {
        "Graph_Irregular_Random": 1 if GRAPH_MAP[graph]==1 else 0,
        "Graph_Spike_Then_Low":   1 if GRAPH_MAP[graph]==2 else 0,
        "Graph_High_Weekends":    1 if GRAPH_MAP[graph]==3 else 0,
        "Graph_Uniform_Daily":    1 if GRAPH_MAP[graph]==0 else 0,
        "Expenditure_Graph_nan":  0,
    }

    # ── NN ────────────────────────────────────────────────────
    nn_row = {
        "Place_Grew_Up":      float(PLACE_MAP[place]),
        "Price_Importance":   float(IMP_MAP[price_imp]),
        "Brand_Importance":   float(IMP_MAP[brand_imp]),
        "Peer_Importance":    float(IMP_MAP[peer_imp]),
        "Utility_Importance": float(IMP_MAP[utility_imp]),
        **{k:float(v) for k,v in track_ohe.items()},
        **{k:float(v) for k,v in graph_ohe.items()},
        **{k:float(v) for k,v in jbs.items()},
        "Unplanned_Purchases": float(up_sc),
        "Peer_Influence":      float(pi_sc),
        "Finance_Confidence":  float(fc_sc),
        **{k:float(v) for k,v in bf.items()},
    }
    nn_df = pd.DataFrame([nn_row])
    for c in M["nn_cols"]:
        if c not in nn_df.columns: nn_df[c] = 0.0
    nn_df = nn_df[M["nn_cols"]].astype(float)

    probs      = M["nn"].predict(nn_df.values, verbose=0)[0]
    spend_tier = int(np.argmax(probs)) + 1
    exp_spend  = float(np.dot(probs, np.arange(1,11)))
    group      = 1 if spend_tier<=3 else (2 if spend_tier<=6 else 3)

    # ── RF Risk Score ─────────────────────────────────────────
    rf_row = {
        "Place_Grew_Up":             place,
        "Price_Importance":          price_imp,
        "Brand_Importance":          brand_imp,
        "Peer_Importance":           peer_imp,
        "Utility_Importance":        utility_imp,
        "Track_Expenditures":        track,
        "Expenditure_Graph":         graph,
        "Justify_Unexpected_Expense":", ".join(justify_sel) if justify_sel else "None",
        "Unplanned_Purchases":       unplanned,
        "Peer_Influence":            peer_inf,
        "Finance_Confidence":        fin_conf,
        "Group":                     str(group),
        **{k:("Yes" if v==1 else "No") for k,v in bf.items()},
    }
    rf_df = pd.DataFrame([rf_row])
    for c in M["rf_cols"]:
        if c not in rf_df.columns: rf_df[c] = 0
    rf_df = rf_df[M["rf_cols"]].copy()

    for col, cat_order in M["rf_factorize"].items():
        if col in rf_df.columns:
            val = str(rf_df[col].values[0])
            rf_df[col] = cat_order.index(val) if val in cat_order else 0
    rf_df = rf_df.apply(pd.to_numeric, errors='coerce').fillna(0)

    rf_scaled     = M["rf_scaler"].transform(rf_df)
    weighted_user = rf_scaled * M["rf_importances"]
    weighted_safe = M["rf_safe_ref"] * M["rf_importances"]
    dist          = float(np.linalg.norm(weighted_user - weighted_safe))
    risk_score    = round(float(np.clip(M["rf_score_scaler"].transform([[dist]])[0][0], 1, 100)), 2)

    # ── KMeans Cluster ────────────────────────────────────────
    km_row = {c: nn_row.get(c, 0.0) for c in M["km_cols"]}
    km_df  = pd.DataFrame([km_row])[M["km_cols"]].astype(float)
    raw_l  = int(M["km"].predict(M["km_scaler"].transform(km_df))[0])
    cluster = M["km_fusion"].get(raw_l, 0)
    cluster = min(cluster, len(PERSONAS)-1)
    persona = PERSONAS[cluster]

    profiles = M["km_profiles"]
    cluster_avg = float(profiles.loc[cluster,"Risk_score"]) if cluster in profiles.index else risk_score

    # ══════════════════════════════════════════════════════════
    # RESULTS
    # ══════════════════════════════════════════════════════════
    st.divider()
    st.header("🧠 Your Financial Intelligence Report")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(f"""<div class="card card-green">
            <h3>📊 Spend Tier</h3>
            <h1 style="color:#2c3e50;">{spend_tier}/10</h1>
            <p>Weighted tier: {exp_spend:.1f}</p>
        </div>""", unsafe_allow_html=True)
    with c2:
        rc = "card-red" if risk_score>=67 else ("card-orange" if risk_score>=34 else "card-green")
        em = "🔴" if risk_score>=67 else ("🟡" if risk_score>=34 else "🟢")
        st.markdown(f"""<div class="card {rc}">
            <h3>{em} Risk Score</h3>
            <h1 style="color:#2c3e50;">{risk_score}</h1>
            <p>Group avg: {cluster_avg:.1f}</p>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="card">
            <h3>🧬 Persona</h3>
            <h2 style="color:#2c3e50;">{persona['emoji']} {persona['name']}</h2>
            <p>Cluster {cluster} of {len(PERSONAS)-1}</p>
        </div>""", unsafe_allow_html=True)

    # Persona info
    st.divider()
    st.subheader(f"{persona['emoji']} Your Profile: {persona['name']}")
    st.info(persona["desc"])

    # vs cluster comparison
    diff = risk_score - cluster_avg
    if diff > 5:
        st.warning(f"⚠️ Your risk score is **{diff:.1f} pts above** your cluster average ({cluster_avg:.1f}) — you're performing worse than most in your group.")
    elif diff < -5:
        st.success(f"✅ Your risk score is **{abs(diff):.1f} pts below** your cluster average ({cluster_avg:.1f}) — you're performing better than most in your group!")
    else:
        st.info(f"📊 Your risk score ({risk_score}) is close to your cluster average ({cluster_avg:.1f}).")

    # Recommendations
    st.divider()
    st.subheader("💡 Recommendations to Reduce Risk")

    if cluster == 0:
        st.success("✅ You're already in the safest group — **Mindful Minimalist**. Keep it up!")
    else:
        target_p = PERSONAS[cluster-1]
        st.markdown(f"**Goal:** Move from {persona['emoji']} **{persona['name']}** → {target_p['emoji']} **{target_p['name']}**")

        recs = []
        for feat, (direction, advice, cost) in FEATURE_ADVICE.items():
            if feat not in profiles.columns or cluster-1 not in profiles.index:
                continue
            target_val = float(profiles.loc[cluster-1, feat])
            user_val   = float(nn_row.get(feat, 0))
            gap        = abs(user_val - target_val)
            if direction == "decrease" and user_val > target_val and gap > 0.05:
                recs.append((gap/cost, advice, feat, user_val, target_val))
            elif direction == "increase" and user_val < target_val and gap > 0.05:
                recs.append((gap/cost, advice, feat, user_val, target_val))

        recs.sort(reverse=True)
        for i, (eff, advice, feat, uv, tv) in enumerate(recs[:3], 1):
            st.markdown(f"**{i}. {advice}**")
            st.caption(f"`{feat}`: your value `{uv:.2f}` → target `{tv:.2f}`")

    # Full journey
    if cluster > 1:
        with st.expander("📍 See your full risk reduction journey"):
            for c_id in range(cluster, 0, -1):
                p_cur = PERSONAS[c_id]
                p_nxt = PERSONAS[c_id-1]
                r_cur = profiles.loc[c_id,"Risk_score"] if c_id in profiles.index else "?"
                r_nxt = profiles.loc[c_id-1,"Risk_score"] if c_id-1 in profiles.index else "?"
                st.markdown(f"**Step {cluster - c_id + 1}:** {p_cur['emoji']} {p_cur['name']} → {p_nxt['emoji']} {p_nxt['name']} *(Risk: ~{r_cur:.0f} → ~{r_nxt:.0f})*")

    st.divider()
    st.caption("Built for academic purposes at MNIT Jaipur. All predictions are model-generated estimates.")
