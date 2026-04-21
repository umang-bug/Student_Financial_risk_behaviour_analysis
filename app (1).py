import streamlit as st
import numpy as np
import pandas as pd
import joblib

st.set_page_config(
    page_title="Student Financial Profiler | MNIT Jaipur",
    page_icon="💰", layout="centered"
)

# ══════════════════════════════════════════════════════════════
# GLOBAL CSS
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

/* ── Force light theme — prevent dark mode text visibility issues ── */
html, body, [data-testid="stAppViewContainer"], .stApp {
    font-family: 'Inter', sans-serif !important;
    background-color: #f0f4f8 !important;
    color: #1a202c !important;
}
p, span, div, label, h1, h2, h3, h4, h5 { color: #1a202c !important; }
.stSlider label, .stSelectbox label, .stRadio label,
.stMultiSelect label, [data-testid="stWidgetLabel"] {
    color: #1a202c !important; font-weight: 600 !important;
}
.stRadio [data-testid="stMarkdownContainer"] p { color: #1a202c !important; }
[data-baseweb="select"] { background: white !important; }
[data-baseweb="select"] * { color: #1a202c !important; }
[data-testid="stExpander"] { background: white !important; border-radius: 12px !important; border: 1px solid #e2e8f0 !important; }
[data-testid="stExpander"] summary { color: #1a202c !important; font-weight: 600 !important; }
[data-testid="stAlert"] { border-radius: 12px !important; }

#MainMenu, footer, header { visibility: hidden; }

.stButton>button {
    background: linear-gradient(135deg, #00c04b, #00a040) !important;
    color: white !important; font-weight: 700 !important;
    font-size: 1rem !important; border-radius: 12px !important;
    border: none !important; padding: 0.65rem 2rem !important;
    box-shadow: 0 4px 15px rgba(0,192,75,0.35) !important;
    transition: all 0.15s ease !important;
}
.stButton>button:hover { transform: translateY(-2px) !important; box-shadow: 0 6px 20px rgba(0,192,75,0.45) !important; }

.metric-card { background: #ffffff !important; border-radius: 16px; padding: 1.4rem 1.6rem; box-shadow: 0 2px 16px rgba(0,0,0,0.07); text-align: center; margin-bottom: 1rem; }
.persona-card { background: #ffffff !important; border-radius: 16px; padding: 1.6rem 2rem; box-shadow: 0 2px 20px rgba(0,0,0,0.08); margin: 1rem 0; }
.rec-card { background: #ffffff !important; border-radius: 12px; padding: 1rem 1.4rem; box-shadow: 0 2px 10px rgba(0,0,0,0.06); margin: 0.5rem 0; border-left: 4px solid #00c04b; }

.scale-hint { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 0.4rem 0.9rem; font-size: 0.78rem; margin: -0.2rem 0 0.8rem 0; display: flex; justify-content: space-between; }
.scale-hint span { color: #64748b !important; }

.section-header { background: linear-gradient(135deg, #1a1a2e, #16213e); color: white !important; border-radius: 12px; padding: 0.9rem 1.5rem; margin: 1.5rem 0 1rem 0; font-size: 1rem; font-weight: 700; }

.step-indicator { display: flex; justify-content: center; gap: 0.6rem; margin: 1rem 0 2rem 0; }
.step-dot { width: 36px; height: 36px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 0.85rem; }
.step-active   { background: #00c04b; color: white !important; box-shadow: 0 0 0 3px rgba(0,192,75,0.3); }
.step-done     { background: #d1fae5; color: #059669 !important; }
.step-inactive { background: #e2e8f0; color: #94a3b8 !important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════
PERSONAS = {
    0: {"name":"Mindful Minimalist",    "emoji":"🌿", "color":"#00c04b",
        "badge":"SAFEST",
        "desc":"Lowest impulse buying and peer influence. High financial confidence. Spends mainly on food with controlled spending bursts. Utility-driven decisions."},
    1: {"name":"Cautious Bank Checker", "emoji":"🏦", "color":"#27ae60",
        "badge":"LOW RISK",
        "desc":"Regularly checks bank balance but no structured budget. Irregular random spending. Mostly food-only budget. Financially aware but not yet disciplined."},
    2: {"name":"Balanced Explorer",     "emoji":"🧭", "color":"#f1c40f",
        "badge":"MODERATE",
        "desc":"The average student — middle of everything. Uses payment apps, weekend + irregular spender. Food & travel focused, justifies trip expenses."},
    3: {"name":"All-In Spender",        "emoji":"🛒", "color":"#e67e22",
        "badge":"ELEVATED",
        "desc":"Spends across ALL budget categories simultaneously. Tracks via apps but still high risk. Heavily discount-driven, justifies trips frequently."},
    4: {"name":"Weekend Party Shopper", "emoji":"🎉", "color":"#e74c3c",
        "badge":"HIGH RISK",
        "desc":"High weekend spending spikes. 100% party-driven expenses. Fashion is top budget. Very high impulse and peer influence. Discount-driven splurger."},
    5: {"name":"Financial Phantom",     "emoji":"👻", "color":"#c0392b",
        "badge":"CRITICAL",
        "desc":"Zero financial tracking. Completely irregular unstructured spending. Entertainment-first. Highest impulse in all clusters."},
}

# Spend tier → approximate monthly amount (₹)
SPEND_TIERS = {1:1500,2:2500,3:3500,4:4500,5:6000,6:7500,7:9000,8:11000,9:13500,10:16000}

# (direction, title, action_tip, cost, your_issue, safer_group_note)
FEATURE_ADVICE = {
    "Unplanned_Purchases": ("decrease","Plan purchases in advance",
        "Avoid opening shopping apps without a clear intent to buy.",2,
        "You tend to make purchases that were not planned in advance.",
        "People in the safer group almost always buy with a clear plan."),
    "Peer_Influence":      ("decrease","Resist peer-driven spending",
        "Set your own monthly budget and stick to it regardless of what friends spend.",2,
        "Your spending is heavily driven by social situations and peer choices.",
        "People in the safer group make spending decisions independently."),
    "Finance_Confidence":  ("increase","Build financial confidence",
        "Start tracking every expense daily for just 30 days to build awareness.",1,
        "You feel uncertain or unconfident about managing your own finances.",
        "People in the safer group actively track and plan their money."),
    "Price_Importance":    ("increase","Compare prices before buying",
        "Check at least 2-3 price options before any purchase above ₹200.",1,
        "You rarely compare prices or look for better deals before buying.",
        "People in the safer group consistently compare prices before spending."),
    "Brand_Importance":    ("decrease","Choose value over brand",
        "Try store-brand or unbranded alternatives for everyday purchases.",2,
        "You tend to choose products based on brand name even when cheaper alternatives exist.",
        "People in the safer group choose based on utility, not brand reputation."),
    "Utility_Importance":  ("increase","Focus on long-term utility",
        "Before buying, honestly ask: will I still be using this 6 months from now?",1,
        "You often buy without fully considering how useful the item will be long-term.",
        "People in the safer group always weigh long-term value before purchasing."),
    "Budget_Fashion":      ("decrease","Cut fashion spending",
        "Limit yourself to one clothing or fashion purchase per month at most.",2,
        "Fashion and clothing is currently a significant part of your monthly budget.",
        "People in the safer group allocate very little budget to fashion."),
    "Budget_Entertainment":("decrease","Reduce entertainment spend",
        "Set a fixed weekly entertainment budget — stop spending on it once reached.",1,
        "Entertainment is taking up a notable portion of your monthly spending.",
        "People in the safer group keep entertainment costs minimal."),
    "Budget_Subscriptions":("decrease","Audit your subscriptions",
        "List every active subscription and cancel anything unused in the past 2 weeks.",1,
        "You are currently spending on multiple subscriptions.",
        "People in the safer group minimise recurring subscription costs."),
    "Discounts(JBS)":      ("decrease","Avoid discount-triggered buying",
        "Remember: a discount is only a saving if you were already going to buy it.",1,
        "You tend to justify unexpected large purchases because of discounts or deals.",
        "People in the safer group do not let discounts trigger unplanned spending."),
    "Party(JBS)":          ("decrease","Control social event spending",
        "At the start of each month, decide a fixed budget for social events and stick to it.",2,
        "You frequently justify large unplanned expenses for social celebrations.",
        "People in the safer group keep social spending within planned limits."),
}

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

# ══════════════════════════════════════════════════════════════
# LOAD MODELS
# ══════════════════════════════════════════════════════════════
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

# ══════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════
if "page" not in st.session_state:
    st.session_state.page = "landing"
if "results" not in st.session_state:
    st.session_state.results = None

# ══════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════
def format_spend(tier_float):
    amount = 0
    for i in range(1, 10):
        w = max(0, 1 - abs(tier_float - i))
        amount += w * SPEND_TIERS[i]
    total = int(amount)
    if total >= 1000:
        return f"₹{total/1000:.1f}k"
    return f"₹{total}"

def risk_meter_svg(score, color):
    """SVG circular risk meter."""
    pct   = score / 100
    r     = 70
    cx, cy = 90, 90
    circ  = 2 * 3.14159 * r
    dash  = pct * circ
    gap   = circ - dash
    # color gradient stops
    c1 = "#00c04b" if score < 34 else ("#f39c12" if score < 67 else "#e74c3c")
    return f"""
    <div style="text-align:center;">
      <svg width="180" height="180" viewBox="0 0 180 180">
        <!-- Background ring -->
        <circle cx="{cx}" cy="{cy}" r="{r}"
                fill="none" stroke="#e2e8f0" stroke-width="14" />
        <!-- Score arc -->
        <circle cx="{cx}" cy="{cy}" r="{r}"
                fill="none" stroke="{c1}" stroke-width="14"
                stroke-linecap="round"
                stroke-dasharray="{dash:.1f} {gap:.1f}"
                transform="rotate(-90 {cx} {cy})" />
        <!-- Score text -->
        <text x="{cx}" y="{cy-8}" text-anchor="middle"
              font-size="28" font-weight="800" fill="#1a202c"
              font-family="Inter,sans-serif">{score:.1f}</text>
        <text x="{cx}" y="{cy+14}" text-anchor="middle"
              font-size="11" fill="#718096"
              font-family="Inter,sans-serif">RISK SCORE</text>
        <text x="{cx}" y="{cy+30}" text-anchor="middle"
              font-size="10" fill="{c1}" font-weight="700"
              font-family="Inter,sans-serif">{"LOW" if score<34 else ("MEDIUM" if score<67 else "HIGH")}</text>
      </svg>
    </div>
    """

def spend_bar_html(tier, exp_spend):
    amount_str = format_spend(exp_spend)
    pct   = (tier / 10) * 100
    color = "#00c04b" if tier <= 3 else ("#f39c12" if tier <= 6 else "#e74c3c")
    label = "Low Spender" if tier <= 3 else ("Moderate Spender" if tier <= 6 else "High Spender")
    tier_dots = "".join([
        f"<div style=\'width:10px;height:10px;border-radius:50%;background:{'"+color+"' if i < tier else '#e2e8f0'};margin:1px;\'></div>"
        for i in range(10)
    ])
    return (
        f"<div style=\'background:#ffffff;border-radius:16px;padding:1.4rem 1.6rem;"
        f"box-shadow:0 2px 16px rgba(0,0,0,0.07);text-align:center;\'>"
        f"<div style=\'font-size:0.7rem;font-weight:700;color:#718096;letter-spacing:1.5px;"
        f"text-transform:uppercase;margin-bottom:0.5rem;\'>Monthly Spend</div>"
        f"<div style=\'font-size:2.6rem;font-weight:800;color:#1a202c;line-height:1;\'>{amount_str}</div>"
        f"<div style=\'display:inline-block;background:{color}22;color:{color};font-size:0.7rem;"
        f"font-weight:700;border-radius:99px;padding:0.2rem 0.7rem;margin:0.4rem 0;\'>{label}</div>"
        f"<div style=\'background:#e2e8f0;border-radius:99px;height:6px;overflow:hidden;margin-top:0.6rem;\'>"
        f"<div style=\'width:{pct}%;height:100%;background:{color};border-radius:99px;\'></div>"
        f"</div>"

        f"</div>"
    )

def step_indicator(current):
    steps = ["Welcome","Survey","Results"]
    html  = '<div class="step-indicator">'
    for i, label in enumerate(steps):
        if i < current:
            cls = "step-done"
            icon = "✓"
        elif i == current:
            cls = "step-active"
            icon = str(i+1)
        else:
            cls = "step-inactive"
            icon = str(i+1)
        html += f'<div style="text-align:center;">'
        html += f'<div class="step-dot {cls}">{icon}</div>'
        html += f'<div style="font-size:0.7rem;margin-top:4px;color:#64748b;">{label}</div>'
        html += '</div>'
    html += '</div>'
    return html

# ══════════════════════════════════════════════════════════════
# PAGE 1: LANDING
# ══════════════════════════════════════════════════════════════
def show_landing():
    st.markdown(step_indicator(0), unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center;padding:2rem 0 1rem 0;">
      <div style="font-size:4rem;margin-bottom:0.5rem;">💰</div>
      <h1 style="font-size:2.4rem;font-weight:800;color:#1a202c;margin:0;">
        Student Financial Profiler
      </h1>
      <p style="font-size:1rem;color:#64748b;margin:0.5rem 0 0 0;">
        MNIT Jaipur &nbsp;·&nbsp; ML-Powered Financial Analysis
      </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background:white;border-radius:20px;padding:2rem;
                box-shadow:0 4px 24px rgba(0,0,0,0.08);margin:1.5rem 0;">
      <h3 style="color:#1a202c;margin-top:0;">What you'll get 🎯</h3>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;">
        <div style="background:#f0fdf4;border-radius:12px;padding:1rem;">
          <div style="font-size:1.5rem;">📊</div>
          <div style="font-weight:600;color:#1a202c;font-size:0.95rem;">Expected Monthly Spend</div>
          <div style="color:#64748b;font-size:0.82rem;">Predicted from your behavior patterns</div>
        </div>
        <div style="background:#fef3c7;border-radius:12px;padding:1rem;">
          <div style="font-size:1.5rem;">🎯</div>
          <div style="font-weight:600;color:#1a202c;font-size:0.95rem;">Financial Risk Score</div>
          <div style="color:#64748b;font-size:0.82rem;">0–100 scale with visual meter</div>
        </div>
        <div style="background:#fdf2f8;border-radius:12px;padding:1rem;">
          <div style="font-size:1.5rem;">🧬</div>
          <div style="font-weight:600;color:#1a202c;font-size:0.95rem;">Behavioral Persona</div>
          <div style="color:#64748b;font-size:0.82rem;">Your spending personality cluster</div>
        </div>
        <div style="background:#eff6ff;border-radius:12px;padding:1rem;">
          <div style="font-size:1.5rem;">💡</div>
          <div style="font-weight:600;color:#1a202c;font-size:0.95rem;">Personalized Tips</div>
          <div style="color:#64748b;font-size:0.82rem;">Actionable steps to improve</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background:linear-gradient(135deg,#1a1a2e,#16213e);border-radius:16px;
                padding:1.2rem 1.6rem;color:white;margin-bottom:1.5rem;font-size:0.88rem;">
      🔒 <strong>Privacy first</strong> — All responses are used only for academic analysis at MNIT Jaipur.
      No personal identifiable data is stored.
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1,2,1])
    with c2:
        if st.button("🚀 Start Assessment", use_container_width=True):
            st.session_state.page = "survey"
            st.rerun()

# ══════════════════════════════════════════════════════════════
# PAGE 2: SURVEY
# ══════════════════════════════════════════════════════════════
def show_survey():
    st.markdown(step_indicator(1), unsafe_allow_html=True)

    st.markdown("""
    <h2 style="text-align:center;font-weight:800;color:#1a202c;margin-bottom:0.2rem;">
        📋 Financial Behavior Survey
    </h2>
    <p style="text-align:center;color:#64748b;margin-bottom:1.5rem;">
        Answer honestly — your results depend on it!
    </p>
    """, unsafe_allow_html=True)

    with st.form("survey_form"):

        # ── Section 1 ────────────────────────────────────────
        st.markdown('<div class="section-header">🏠 Section 1 — Background</div>', unsafe_allow_html=True)
        place = st.radio("Where did you grow up?", PLACE_OPTS, horizontal=True)

        # ── Section 2 ────────────────────────────────────────
        st.markdown('<div class="section-header">💸 Section 2 — Spending Behavior</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            unplanned = st.slider("🛍️ Frequency of unplanned purchases", 1, 5, 3)
            st.markdown('<div class="scale-hint"><span>1 — I never buy without planning</span><span>5 — I buy impulsively very often</span></div>', unsafe_allow_html=True)
            fin_conf  = st.slider("💪 Financial management confidence", 1, 5, 3)
            st.markdown('<div class="scale-hint"><span>1 — I have no idea about my finances</span><span>5 — I manage money very well</span></div>', unsafe_allow_html=True)
        with col2:
            peer_inf  = st.slider("👥 Peer pressure influence on spending", 1, 5, 3)
            st.markdown('<div class="scale-hint"><span>1 — Friends never affect my spending</span><span>5 — I always spend to match friends</span></div>', unsafe_allow_html=True)

        # ── Section 3 ────────────────────────────────────────
        st.markdown('<div class="section-header">🎯 Section 3 — Purchase Priorities</div>', unsafe_allow_html=True)
        st.markdown('<div style="background:#fffbeb;border:1px solid #fde68a;border-radius:8px;padding:0.6rem 1rem;font-size:0.82rem;color:#92400e;margin-bottom:0.8rem;">💡 Rate how much each factor influences your purchasing decision</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            price_imp   = st.select_slider("💲 Price / Cost — Does price heavily influence what you buy?", IMP_OPTS, value="Slightly important")
            peer_imp    = st.select_slider("👥 Peer Recommendation — Do you buy based on what friends suggest?", IMP_OPTS, value="Not important")
        with col2:
            brand_imp   = st.select_slider("🏷️ Brand Reputation — Do you prefer branded products even if costly?", IMP_OPTS, value="Slightly important")
            utility_imp = st.select_slider("🔧 Long-term Utility — Do you consider how long you will use it?", IMP_OPTS, value="Very important")

        # ── Section 4 ────────────────────────────────────────
        st.markdown('<div class="section-header">📈 Section 4 — Tracking & Patterns</div>', unsafe_allow_html=True)
        track = st.selectbox("📱 How do you track monthly expenditures?", TRACK_OPTS)
        graph = st.radio("📉 Which spending pattern best matches yours?", GRAPH_OPTS, horizontal=False)

        # ── Section 5 ────────────────────────────────────────
        st.markdown('<div class="section-header">🛒 Section 5 — Budget & Scenarios</div>', unsafe_allow_html=True)
        budget_sel  = st.multiselect("💰 Where do you spend most of your budget?", BUDGET_OPTS)
        justify_sel = st.multiselect("❓ When would you justify an unexpected ₹1,500+ expense?", JUSTIFY_OPTS)

        c1, c2, c3 = st.columns([1,2,1])
        with c2:
            submitted = st.form_submit_button("🔍 Analyse My Finances", use_container_width=True)

    if submitted:
        with st.spinner("🧠 Running ML models..."):
            results = predict(place, unplanned, peer_inf, fin_conf,
                              price_imp, brand_imp, peer_imp, utility_imp,
                              track, graph, budget_sel, justify_sel)
        st.session_state.results = results
        st.session_state.page    = "results"
        st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("← Back to Welcome"):
        st.session_state.page = "landing"
        st.rerun()

# ══════════════════════════════════════════════════════════════
# PREDICTION ENGINE
# ══════════════════════════════════════════════════════════════
def predict(place, unplanned, peer_inf, fin_conf,
            price_imp, brand_imp, peer_imp, utility_imp,
            track, graph, budget_sel, justify_sel):

    IMP_MAP   = {"Not important":0,"Slightly important":1,"Very important":2}
    PLACE_MAP = {"🏙️ Big metro city":0,"🏢 Medium-sized city":1,"🏘️ Small town":2,"🌾 Rural area":3}
    TRACK_MAP = {
        "I do not keep the track":0,
        "I check my bank balance occasionally.":1,
        "I review my history within payment apps (e.g., UPI, Paytm).":2,
        "I use a dedicated expense-tracking app or spreadsheet.":3,
    }
    GRAPH_MAP = {
        "Uniform Daily Expenses":0,"Irregular and Random Spending":1,
        "Spend a lot once and then low spending for rest":2,"Steady Weekdays with High Weekends":3,
    }

    scaled       = M["mms"].transform([[unplanned, peer_inf, fin_conf]])[0]
    up_sc, pi_sc, fc_sc = scaled

    bf = {
        "Budget_FoodDining":    1 if "Food & Dining"                           in budget_sel else 0,
        "Budget_Travel":        1 if "Travel"                                  in budget_sel else 0,
        "Budget_Fashion":       1 if "Fashion"                                 in budget_sel else 0,
        "Budget_Subscriptions": 1 if "Subscriptions (Netflix, Spotify, etc.)" in budget_sel else 0,
        "Budget_Entertainment": 1 if "Fun & Entertainment"                     in budget_sel else 0,
    }
    jbs = {
        "Emergency(JBS)": 1 if "Emergencies"       in " ".join(justify_sel) else 0,
        "Discounts(JBS)": 1 if "discount"           in " ".join(justify_sel).lower() else 0,
        "Party(JBS)":     1 if "celebrations"       in " ".join(justify_sel).lower() else 0,
        "Workshop(JBS)":  1 if "Skill development"  in " ".join(justify_sel) else 0,
        "Trip(JBS)":      1 if "planned trip"        in " ".join(justify_sel).lower() else 0,
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

    # NN
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

    # RF Risk
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

    # Assign cluster by finding closest avg risk to user's RF risk score
    # This keeps cluster and risk score always consistent
    profiles = M["km_profiles"]
    cluster  = min(profiles["Risk_score"].to_dict(),
                   key=lambda c: abs(profiles.loc[c,"Risk_score"] - risk_score))
    cluster  = min(int(cluster), len(PERSONAS)-1)
    cluster_avg = float(profiles.loc[cluster,"Risk_score"])

    return {
        "spend_tier": spend_tier, "exp_spend": exp_spend,
        "risk_score": risk_score, "cluster": cluster,
        "cluster_avg": cluster_avg, "profiles": profiles,
        "nn_row": nn_row,
    }

# ══════════════════════════════════════════════════════════════
# PAGE 3: RESULTS
# ══════════════════════════════════════════════════════════════
def show_results():
    r = st.session_state.results
    spend_tier  = r["spend_tier"]
    exp_spend   = r["exp_spend"]
    risk_score  = r["risk_score"]
    cluster     = r["cluster"]
    cluster_avg = r["cluster_avg"]
    profiles    = r["profiles"]
    nn_row      = r["nn_row"]
    persona     = PERSONAS[cluster]

    st.markdown(step_indicator(2), unsafe_allow_html=True)

    st.markdown(f"""
    <h2 style="text-align:center;font-weight:800;color:#1a202c;margin-bottom:0.3rem;">
        🧠 Your Financial Intelligence Report
    </h2>
    <p style="text-align:center;color:#64748b;margin-bottom:2rem;">
        Powered by Neural Network · Random Forest · KMeans Clustering
    </p>
    """, unsafe_allow_html=True)

    # ── 3 metric cards ───────────────────────────────────────
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(spend_bar_html(spend_tier, exp_spend), unsafe_allow_html=True)

    with c2:
        risk_color = "#00c04b" if risk_score<34 else ("#f39c12" if risk_score<67 else "#e74c3c")
        st.markdown(risk_meter_svg(risk_score, risk_color), unsafe_allow_html=True)

    with c3:
        diff   = risk_score - cluster_avg
        d_sign = "▲" if diff > 0 else "▼"
        d_col  = "#e74c3c" if diff > 5 else ("#00c04b" if diff < -5 else "#718096")
        st.markdown(f"""
        <div style="background:white;border-radius:16px;padding:1.4rem 1.6rem;
                    box-shadow:0 2px 16px rgba(0,0,0,0.07);text-align:center;height:100%;">
          <div style="font-size:0.75rem;font-weight:700;color:#718096;letter-spacing:1px;
                      text-transform:uppercase;margin-bottom:0.4rem;">Your Group</div>
          <div style="font-size:2.8rem;">{persona['emoji']}</div>
          <div style="font-size:0.95rem;font-weight:700;color:#1a202c;">{persona['name']}</div>
          <div style="display:inline-block;background:{persona['color']}22;color:{persona['color']};
                      font-size:0.7rem;font-weight:700;border-radius:99px;
                      padding:0.2rem 0.8rem;margin-top:0.4rem;">{persona['badge']}</div>
          <div style="margin-top:0.8rem;font-size:0.82rem;color:{d_col};font-weight:600;">
            {d_sign} {abs(diff):.1f} pts vs group avg ({cluster_avg:.1f})
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Persona description ───────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class="persona-card" style="border-top:4px solid {persona['color']};">
      <div style="display:flex;align-items:center;gap:1rem;margin-bottom:0.8rem;">
        <span style="font-size:2.5rem;">{persona['emoji']}</span>
        <div>
          <div style="font-size:1.3rem;font-weight:800;color:#1a202c;">{persona['name']}</div>
          <div style="display:inline-block;background:{persona['color']}22;color:{persona['color']};
                      font-size:0.75rem;font-weight:700;border-radius:99px;padding:0.2rem 0.8rem;">
            {persona['badge']} · Cluster {cluster}</div>
        </div>
      </div>
      <p style="color:#4a5568;line-height:1.7;margin:0;">{persona['desc']}</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Comparison vs group ───────────────────────────────────
    diff = risk_score - cluster_avg
    if diff > 5:
        st.warning(f"⚠️ Your risk score is **{diff:.1f} pts higher** than your group average ({cluster_avg:.1f}) — you're at more risk than most in your cluster.")
    elif diff < -5:
        st.success(f"✅ Your risk score is **{abs(diff):.1f} pts lower** than your group average ({cluster_avg:.1f}) — you're doing better than most in your cluster!")
    else:
        st.info(f"📊 Your risk score ({risk_score}) is in line with your group average ({cluster_avg:.1f}).")

    # ── Recommendations ───────────────────────────────────────
    st.markdown('<div class="section-header">💡 Personalized Action Plan</div>', unsafe_allow_html=True)

    if cluster == 0:
        st.markdown(f"""
        <div class="rec-card" style="background:#f0fdf4;border-left-color:#00c04b;">
          <div style="font-size:1.2rem;font-weight:700;color:#00c04b;">
            🎉 You're already in the safest group!
          </div>
          <div style="color:#4a5568;margin-top:0.4rem;">
            Keep maintaining your disciplined financial habits.
            You're a <strong>Mindful Minimalist</strong> — a role model for financial health.
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        target_p = PERSONAS[cluster - 1]
        st.markdown(f"""
        <div style="background:#eff6ff;border-radius:12px;padding:1rem 1.4rem;margin-bottom:1rem;">
          <strong>🎯 Goal:</strong> Move from {persona['emoji']} <strong>{persona['name']}</strong>
          → {target_p['emoji']} <strong>{target_p['name']}</strong>
          &nbsp;|&nbsp; Risk drop: ~{cluster_avg:.0f} → ~{(profiles.loc[cluster-1,'Risk_score'] if cluster-1 in profiles.index else cluster_avg-10):.0f}
        </div>
        """, unsafe_allow_html=True)

        # build recs
        recs = []
        for feat, advice_tuple in FEATURE_ADVICE.items():
            direction, title, action_tip, cost, your_issue, safer_note = advice_tuple
            if feat not in profiles.columns or cluster-1 not in profiles.index:
                continue
            target_val = float(profiles.loc[cluster-1, feat])
            user_val   = float(nn_row.get(feat, 0))
            gap        = abs(user_val - target_val)
            if direction == "decrease" and user_val > target_val and gap > 0.05:
                recs.append((gap/cost, title, action_tip, your_issue, safer_note, feat))
            elif direction == "increase" and user_val < target_val and gap > 0.05:
                recs.append((gap/cost, title, action_tip, your_issue, safer_note, feat))

        recs.sort(reverse=True)
        PRIORITY_COLORS = ["#e74c3c","#f39c12","#3498db"]
        PRIORITY_LABELS = ["High Priority","Medium Priority","Low Priority"]

        for i, (eff, title, action_tip, your_issue, safer_note, feat) in enumerate(recs[:3]):
            col = PRIORITY_COLORS[i]
            lbl = PRIORITY_LABELS[i]
            st.markdown(f"""
            <div class="rec-card" style="border-left-color:{col};margin-bottom:0.8rem;">
              <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:0.6rem;">
                <div style="font-weight:700;color:#1a202c;font-size:1rem;">{i+1}. {title}</div>
                <div style="background:{col}22;color:{col};font-size:0.7rem;font-weight:700;
                            border-radius:99px;padding:0.25rem 0.7rem;white-space:nowrap;margin-left:1rem;">
                  {lbl}
                </div>
              </div>
              <div style="background:#fff8f0;border-radius:8px;padding:0.5rem 0.8rem;margin-bottom:0.5rem;font-size:0.85rem;color:#92400e;">
                ⚠️ <strong>Your pattern:</strong> {your_issue}
              </div>
              <div style="background:#f0fdf4;border-radius:8px;padding:0.5rem 0.8rem;margin-bottom:0.5rem;font-size:0.85rem;color:#166534;">
                ✅ <strong>Safer group:</strong> {safer_note}
              </div>
              <div style="font-size:0.85rem;color:#3b82f6;margin-top:0.4rem;">
                💡 <strong>What to do:</strong> {action_tip}
              </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Full journey in expander ──────────────────────────────
    if cluster > 0:
        with st.expander("📍 View Your Full Risk Reduction Journey", expanded=False):
            st.markdown(f"""
            <p style="color:#64748b;font-size:0.9rem;">
              Your complete path from current persona to the safest group.
              Each step represents a meaningful behavioral shift.
            </p>
            """, unsafe_allow_html=True)

            for c_id in range(cluster, -1, -1):
                p       = PERSONAS[c_id]
                r_val   = profiles.loc[c_id, "Risk_score"] if c_id in profiles.index else 0
                is_curr = c_id == cluster
                is_goal = c_id == 0
                bg      = "#f0fdf4" if is_goal else ("#eff6ff" if is_curr else "white")
                border  = "2px solid " + p["color"] if (is_curr or is_goal) else "1px solid #e2e8f0"
                here_badge = "&nbsp;<span style='background:#eff6ff;color:#3b82f6;font-size:0.7rem;padding:0.1rem 0.5rem;border-radius:99px;'>YOU ARE HERE</span>" if is_curr else ""
                goal_badge = "&nbsp;<span style='background:#f0fdf4;color:#00c04b;font-size:0.7rem;padding:0.1rem 0.5rem;border-radius:99px;'>🏁 GOAL</span>" if is_goal else ""
                arrow_html = "<div style='text-align:center;color:#94a3b8;font-size:1.2rem;margin:-0.1rem 0;'>↑</div>" if c_id > 0 else ""
                card_html  = (
                    f"<div style='background:{bg};border:{border};border-radius:12px;"
                    f"padding:0.8rem 1.2rem;margin:0.4rem 0;display:flex;align-items:center;gap:1rem;'>"
                    f"<span style='font-size:1.8rem;'>{p['emoji']}</span>"
                    f"<div style='flex:1;'>"
                    f"<div style='font-weight:700;color:#1a202c;'>{p['name']}{here_badge}{goal_badge}</div>"
                    f"<div style='color:#64748b;font-size:0.82rem;'>Avg Risk: ~{r_val:.0f}</div>"
                    f"</div>"
                    f"<div style='background:{p['color']}22;color:{p['color']};font-size:0.75rem;"
                    f"font-weight:700;border-radius:99px;padding:0.25rem 0.8rem;'>{p['badge']}</div>"
                    f"</div>{arrow_html}"
                )
                st.markdown(card_html, unsafe_allow_html=True)

    # ── Footer ────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1,2,1])
    with c2:
        if st.button("🔄 Take Survey Again", use_container_width=True):
            st.session_state.page    = "survey"
            st.session_state.results = None
            st.rerun()

    st.markdown("""
    <div style="text-align:center;color:#94a3b8;font-size:0.78rem;margin-top:2rem;padding:1rem;">
      Built for academic purposes at MNIT Jaipur &nbsp;·&nbsp; All predictions are ML-generated estimates
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# ROUTER
# ══════════════════════════════════════════════════════════════
page = st.session_state.page
if   page == "landing":  show_landing()
elif page == "survey":   show_survey()
elif page == "results":  show_results()
