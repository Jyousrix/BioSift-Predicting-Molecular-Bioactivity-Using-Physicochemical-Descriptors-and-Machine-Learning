import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ─────────────────────────────────────────────────────────────────────────────
# 1. Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="BioSift", layout="wide")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Theme detection
# ─────────────────────────────────────────────────────────────────────────────
try:
    theme_param = st.query_params.get("theme", "dark")
except Exception:
    theme_param = "dark"
is_dark = (theme_param != "light")

if is_dark:
    BG            = "#0B0F12"
    SURFACE       = "#131920"
    BORDER        = "#1E2A35"
    H1_COLOR      = "#F8FAFC"
    DESC_COLOR    = "#CBD5E1"
    NOTICE_COLOR  = "#94A3B8"
    DIVIDER       = "#1E2A35"
    LABEL_COLOR   = "#CBD5E1"
    INPUT_BG      = "#131920"
    SECTION_HEAD  = "#F8FAFC"
    BODY_TEXT     = "#E2E8F0"
    METRIC_BG     = "#131920"
    MOL_OPACITY   = "0.04"
    MOL_COLOR     = "#10B981"
    BTN_TEXT      = "#0B0F12"
else:
    BG            = "#F8FAFC"
    SURFACE       = "#FFFFFF"
    BORDER        = "#E2E8F0"
    H1_COLOR      = "#0F172A"
    DESC_COLOR    = "#1E293B"
    NOTICE_COLOR  = "#334155"
    DIVIDER       = "#E2E8F0"
    LABEL_COLOR   = "#0F172A"
    INPUT_BG      = "#FFFFFF"
    SECTION_HEAD  = "#0F172A"
    BODY_TEXT     = "#0F172A"
    METRIC_BG     = "#FFFFFF"
    MOL_OPACITY   = "0.035"
    MOL_COLOR     = "#6B7280"
    BTN_TEXT      = "#FFFFFF"

# ─────────────────────────────────────────────────────────────────────────────
# 3. CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
.stApp {{
    background-color: {BG} !important;
}}
.stApp::before {{
    content: '';
    position: fixed;
    inset: 0;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='900' height='900' opacity='{MOL_OPACITY}'%3E%3Ccircle cx='120' cy='200' r='18' fill='{MOL_COLOR.replace('#','%23')}'/%3E%3Ccircle cx='300' cy='120' r='14' fill='{MOL_COLOR.replace('#','%23')}'/%3E%3Ccircle cx='480' cy='250' r='18' fill='{MOL_COLOR.replace('#','%23')}'/%3E%3Ccircle cx='650' cy='150' r='14' fill='{MOL_COLOR.replace('#','%23')}'/%3E%3Ccircle cx='780' cy='300' r='18' fill='{MOL_COLOR.replace('#','%23')}'/%3E%3Ccircle cx='200' cy='420' r='14' fill='{MOL_COLOR.replace('#','%23')}'/%3E%3Ccircle cx='400' cy='500' r='18' fill='{MOL_COLOR.replace('#','%23')}'/%3E%3Ccircle cx='600' cy='420' r='14' fill='{MOL_COLOR.replace('#','%23')}'/%3E%3Ccircle cx='750' cy='550' r='18' fill='{MOL_COLOR.replace('#','%23')}'/%3E%3Ccircle cx='100' cy='650' r='14' fill='{MOL_COLOR.replace('#','%23')}'/%3E%3Ccircle cx='320' cy='700' r='18' fill='{MOL_COLOR.replace('#','%23')}'/%3E%3Ccircle cx='520' cy='660' r='14' fill='{MOL_COLOR.replace('#','%23')}'/%3E%3Ccircle cx='820' cy='750' r='18' fill='{MOL_COLOR.replace('#','%23')}'/%3E%3Cline x1='120' y1='200' x2='300' y2='120' stroke='{MOL_COLOR.replace('#','%23')}' stroke-width='2'/%3E%3Cline x1='300' y1='120' x2='480' y2='250' stroke='{MOL_COLOR.replace('#','%23')}' stroke-width='2'/%3E%3Cline x1='480' y1='250' x2='650' y2='150' stroke='{MOL_COLOR.replace('#','%23')}' stroke-width='2'/%3E%3Cline x1='650' y1='150' x2='780' y2='300' stroke='{MOL_COLOR.replace('#','%23')}' stroke-width='2'/%3E%3Cline x1='120' y1='200' x2='200' y2='420' stroke='{MOL_COLOR.replace('#','%23')}' stroke-width='2'/%3E%3Cline x1='480' y1='250' x2='400' y2='500' stroke='{MOL_COLOR.replace('#','%23')}' stroke-width='2'/%3E%3Cline x1='780' y1='300' x2='750' y2='550' stroke='{MOL_COLOR.replace('#','%23')}' stroke-width='2'/%3E%3Cline x1='200' y1='420' x2='400' y2='500' stroke='{MOL_COLOR.replace('#','%23')}' stroke-width='2'/%3E%3Cline x1='400' y1='500' x2='600' y2='420' stroke='{MOL_COLOR.replace('#','%23')}' stroke-width='2'/%3E%3Cline x1='600' y1='420' x2='750' y2='550' stroke='{MOL_COLOR.replace('#','%23')}' stroke-width='2'/%3E%3Cline x1='200' y1='420' x2='100' y2='650' stroke='{MOL_COLOR.replace('#','%23')}' stroke-width='2'/%3E%3Cline x1='400' y1='500' x2='320' y2='700' stroke='{MOL_COLOR.replace('#','%23')}' stroke-width='2'/%3E%3Cline x1='600' y1='420' x2='520' y2='660' stroke='{MOL_COLOR.replace('#','%23')}' stroke-width='2'/%3E%3Cline x1='750' y1='550' x2='820' y2='750' stroke='{MOL_COLOR.replace('#','%23')}' stroke-width='2'/%3E%3C/svg%3E");
    background-repeat: repeat;
    background-size: 900px 900px;
    pointer-events: none;
    z-index: 0;
}}

section[data-testid="stSidebar"] {{
    background-color: {SURFACE} !important;
    border-right: 1px solid {BORDER} !important;
}}
section[data-testid="stSidebar"] * {{ color: {BODY_TEXT} !important; }}

h1, .stTitle {{ color: {H1_COLOR} !important; }}
h2, h3       {{ color: {SECTION_HEAD} !important; }}
p, li, label, .stMarkdown p {{ color: {BODY_TEXT} !important; }}

/* ── Tabs ── */
button[data-baseweb="tab"] {{
    color: #64748B !important;
    transition: color 0.2s ease;
}}
button[data-baseweb="tab"][aria-selected="true"],
button[data-baseweb="tab"]:hover,
button[data-baseweb="tab"]:focus {{
    color: #f1c40f !important;
}}
[data-baseweb="tabs"]          {{ --primary: #f1c40f !important; }}
[data-baseweb="tab-highlight"] {{ background-color: #f1c40f !important; }}
[data-baseweb="tab-border"]    {{ background-color: {BORDER} !important; }}

/* ── Metrics ── */
div[data-testid="stMetricBlock"] {{
    background-color: {METRIC_BG} !important;
    border: 1px solid {BORDER} !important;
    border-left: 4px solid #10B981 !important;
    border-radius: 10px;
    padding: 15px;
}}
div[data-testid="stMetricValue"] {{ font-weight: bold; color: {H1_COLOR} !important; }}
div[data-testid="stMetricLabel"] {{ color: {DESC_COLOR} !important; }}

/* ── Number inputs — always editable, text visible ── */
div[data-testid="stNumberInput"] label {{
    color: {LABEL_COLOR} !important;
    transition: color 0.2s;
}}
div[data-testid="stNumberInput"]:hover label,
div[data-testid="stNumberInput"]:focus-within label {{
    color: #f1c40f !important;
}}
div[data-testid="stNumberInput"] input {{
    background-color: {INPUT_BG} !important;
    color: {H1_COLOR} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 6px !important;
    -webkit-text-fill-color: {H1_COLOR} !important;
    opacity: 1 !important;
}}
div[data-testid="stNumberInput"] button {{
    background-color: {SURFACE} !important;
    color: {H1_COLOR} !important;
    border: 1px solid {BORDER} !important;
}}
div[data-testid="stNumberInput"] button:hover {{
    background-color: #10B981 !important;
    color: #ffffff !important;
}}

/* ── All main action buttons ── */
.stButton > button {{
    background-color: #10B981 !important;
    color: {BTN_TEXT} !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 10px 24px !important;
    font-size: 1rem !important;
    transition: background-color 0.2s ease, transform 0.1s ease;
    -webkit-text-fill-color: {BTN_TEXT} !important;
}}
.stButton > button:hover {{
    background-color: #059669 !important;
    transform: translateY(-1px);
}}

/* ── Reset button — distinct style ── */
[data-testid="stHorizontalBlock"] > div:last-child .stButton > button {{
    background-color: {SURFACE} !important;
    color: {BODY_TEXT} !important;
    -webkit-text-fill-color: {BODY_TEXT} !important;
    border: 1px solid {BORDER} !important;
    font-weight: 500 !important;
}}
[data-testid="stHorizontalBlock"] > div:last-child .stButton > button:hover {{
    border-color: #EF4444 !important;
    color: #EF4444 !important;
    -webkit-text-fill-color: #EF4444 !important;
    transform: none;
}}

hr {{ border-color: {DIVIDER} !important; opacity: 1 !important; }}

div[data-testid="stExpander"] {{
    border: 1px solid {BORDER} !important;
    border-radius: 8px !important;
    background-color: {SURFACE} !important;
}}
div[data-testid="stExpander"] summary {{ color: {SECTION_HEAD} !important; transition: color 0.2s; }}
div[data-testid="stExpander"] summary:hover {{ color: #f1c40f !important; }}
div[data-testid="stExpander"] p {{ color: {BODY_TEXT} !important; }}

button[data-testid="stCopyButton"] {{ display: none !important; }}

/* ── Scroll hint ── */
@keyframes slideInFade {{
    0%   {{ opacity: 0; transform: translateY(-45%) scale(0.95); }}
    100% {{ opacity: 1; transform: translateY(-50%) scale(1); }}
}}
#scroll-hint {{
    position: fixed;
    left: 24px;
    top: 50%;
    transform: translateY(-50%);
    background-color: {SURFACE};
    border: 1px solid {BORDER};
    border-left: 4px solid #10B981;
    border-radius: 8px;
    padding: 14px 18px;
    font-size: 0.82rem;
    color: {H1_COLOR};
    font-weight: 600;
    z-index: 99999;
    max-width: 190px;
    line-height: 1.45;
    box-shadow: 0 10px 25px -5px rgba(0,0,0,0.3);
    animation: slideInFade 0.4s cubic-bezier(0.16,1,0.3,1) forwards;
    transition: opacity 0.4s ease;
}}

/* ── Landing ── */
.landing-wrap {{
    max-width: 680px;
    margin: 0 auto;
    padding: 80px 24px 20px 24px;
    text-align: center;
    position: relative;
    z-index: 1;
}}
.landing-title {{
    font-size: 3.4rem;
    font-weight: 800;
    color: #10B981;
    letter-spacing: -1.5px;
    margin-bottom: 20px;
    line-height: 1.1;
}}
.landing-desc {{
    font-size: 1.05rem;
    color: {DESC_COLOR};
    line-height: 1.85;
    margin-bottom: 36px;
}}
.landing-notice-wrap {{
    max-width: 560px;
    margin: 0 auto;
    padding: 0 24px 60px 24px;
    text-align: center;
}}
.landing-divider {{
    border: none;
    border-top: 1px solid {DIVIDER};
    margin: 28px auto 28px auto;
    width: 100%;
}}
.inline-notice {{
    font-size: 0.87rem;
    color: {NOTICE_COLOR};
    line-height: 1.75;
    font-style: italic;
}}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# 4. Session state
# ─────────────────────────────────────────────────────────────────────────────
if "page"             not in st.session_state: st.session_state.page             = "landing"
if "show_scroll_hint" not in st.session_state: st.session_state.show_scroll_hint = False
if "screening_run"    not in st.session_state: st.session_state.screening_run    = False
if "last_result"      not in st.session_state: st.session_state.last_result      = None

# ─────────────────────────────────────────────────────────────────────────────
# 5. LANDING PAGE
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.page == "landing":

    st.markdown("""
        <div class="landing-wrap">
            <div class="landing-title">BioSift</div>
            <div class="landing-desc">
                A descriptor-based machine learning pipeline for predicting small molecule
                bioactivity potential. Input physicochemical descriptors to classify compounds
                as <span style="color:#ffffff; font-weight:600;">Active</span> or
                <span style="color:#ffffff; font-weight:600;">Inactive</span>
                against target cancer cell lines.
            </div>
        </div>
    """, unsafe_allow_html=True)

    col_l, col_c, col_r = st.columns([2, 1, 2])
    with col_c:
        if st.button("Let's Start", use_container_width=True):
            st.session_state.page = "predictor"
            st.rerun()

    st.markdown(f"""
        <div class="landing-notice-wrap">
            <hr class="landing-divider"/>
            <div class="inline-notice">
                <strong style="color:{NOTICE_COLOR}; font-style:normal; letter-spacing:1px; font-size:0.72rem; text-transform:uppercase;">Please Note</strong><br/><br/>
                BioSift utilizes a continuously expanding screening library. While gaps may
                exist within less-studied chemical spaces for this version, we regularly
                integrate new experimental data from the NCI bioassays to refine future
                predictive models. We appreciate your patience as we expand our dataset
                coverage.
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# 6. Load ML Assets
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_assets():
    model    = joblib.load('bioactivity_model.pkl')
    features = joblib.load('features.pkl')
    return model, features

try:
    model, features = load_assets()
    st.sidebar.success("Core ML Engine Online")
except Exception as e:
    st.sidebar.error(f"Asset Loading Failed: {e}")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# 7. Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"<h1 style='color:{H1_COLOR}; font-size:2rem; font-weight:800; margin-bottom:4px;'>BioSift — Screening Portal</h1>", unsafe_allow_html=True)
st.markdown(f"<p style='color:{DESC_COLOR}; margin-top:0; margin-bottom:20px;'>Input your compound's molecular descriptors below to evaluate its bioactivity potential.</p>", unsafe_allow_html=True)
st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────────
# 8. Model Diagnostics sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar.expander("Model Diagnostics", expanded=False):
    try:
        test_proba = model.predict_proba([[1155.46, 0.85, 9, 24, 342.0]])[0]
        st.write(f"Active prob (A-1): `{test_proba[1]:.4f}`")
        st.write(f"class_weight: `{model.class_weight}`")
        st.write(f"n_estimators: `{model.n_estimators}`")
    except Exception as ex:
        st.error(f"Diagnostic error: {ex}")

# ─────────────────────────────────────────────────────────────────────────────
# 9. Compound Library
# ─────────────────────────────────────────────────────────────────────────────
preset_compounds = {
    "Manual Entry (Blank)": {
        "values": {"MolWt": 0.0, "LogP": 0.0, "NumHDonors": 0, "NumHAcceptors": 0, "TPSA": 0.0},
        "name":   "Custom Structural Ligand Scaffold",
        "fact":   "You are evaluating a custom compound. The Active/Inactive result reflects "
                  "how closely your descriptor profile matches known bioactive patterns in the "
                  "NCI training dataset. Compounds with MolWt under 500, LogP between 0-5, "
                  "fewer than 5 H-bond donors and fewer than 10 acceptors tend to have the "
                  "strongest drug-like profiles."
    },
    "Trial Compound A-1  |  CID 44415057": {
        "values": {"MolWt": 1155.46, "LogP": 0.85, "NumHDonors": 9, "NumHAcceptors": 24, "TPSA": 342.0},
        "name":   "Depsipeptide Macrocyclic Antibiotic Scaffold",
        "fact":   "**Depsipeptide Macrocyclic Antibiotic Scaffold (CID 44415057)**\n\n"
                  "This compound belongs to the depsipeptide family — cyclic peptides where "
                  "one or more amide bonds are replaced by ester linkages. With a molecular "
                  "weight exceeding 1,100 Da and an NCI activity score of 100/100, it is one "
                  "of the most potent hits in this dataset. Its dense hydrogen-bonding network "
                  "(9 donors, 24 acceptors) wraps tightly around target proteins like a molecular "
                  "vice-grip, completely locking enzymatic function. Similar scaffolds underpin "
                  "approved antibiotics including daptomycin."
    },
    "Trial Compound A-2  |  CID 6197": {
        "values": {"MolWt": 253.34, "LogP": 0.62, "NumHDonors": 3, "NumHAcceptors": 4, "TPSA": 83.9},
        "name":   "Cyclohexanone-Lactam Anticancer Natural Product",
        "fact":   "**Cyclohexanone-Lactam Anticancer Natural Product (CID 6197)**\n\n"
                  "This compact bicyclic compound (MW 253) satisfies all five of Lipinski's "
                  "rules, making it an excellent oral drug candidate. Its low TPSA (83.9 "
                  "Angstrom squared) permits passive diffusion across cell membranes, while "
                  "its moderate LogP (0.62) balances aqueous solubility with permeability. "
                  "Natural products sharing this skeleton have inspired several anti-tumour "
                  "agents targeting cell cycle regulation. NCI activity score: 55/100."
    },
    "Trial Compound A-3  |  CID 2154": {
        "values": {"MolWt": 441.40, "LogP": -1.85, "NumHDonors": 5, "NumHAcceptors": 10, "TPSA": 206.8},
        "name":   "Folate-Derived Antifolate Antimetabolite (Pemetrexed-Class)",
        "fact":   "**Folate-Derived Antifolate Antimetabolite — CID 2154**\n\n"
                  "Structurally related to Pemetrexed (Alimta), an FDA-approved chemotherapy "
                  "for lung cancer and mesothelioma. Its very negative LogP (-1.85) means it "
                  "must be actively transported into cancer cells via folate receptors, which "
                  "are dramatically overexpressed in tumour tissue. Once inside, it inhibits "
                  "dihydrofolate reductase, starving the cell of nucleotide building blocks "
                  "needed for DNA replication. NCI activity score: 61/100."
    },
    "Trial Compound I-1  |  CID 11122": {
        "values": {"MolWt": 122.12, "LogP": 0.95, "NumHDonors": 0, "NumHAcceptors": 2, "TPSA": 34.1},
        "name":   "2,6-Dimethyl-1,4-Benzoquinone (Inactive Quinone Fragment)",
        "fact":   "**2,6-Dimethyl-1,4-Benzoquinone — CID 11122**\n\n"
                  "At only 122 Da, this tiny quinone fragment is far too small to engage "
                  "meaningfully with a biological receptor binding pocket. While quinone "
                  "scaffolds appear in bioactive natural products such as Vitamin K and "
                  "Coenzyme Q10, this unsubstituted version lacks the extended aromatic "
                  "system and functional groups needed for selective target binding. "
                  "NCI activity score: 10/100."
    },
    "Trial Compound I-2  |  CID 219128": {
        "values": {"MolWt": 364.27, "LogP": 5.32, "NumHDonors": 0, "NumHAcceptors": 0, "TPSA": 0.0},
        "name":   "Iodoalkyl Triphenyl Lipophilic Scaffold (Inactive)",
        "fact":   "**Iodoalkyl Triphenyl Lipophilic Scaffold — CID 219128**\n\n"
                  "With a LogP of 5.32 and a TPSA of exactly 0 Angstrom squared, this "
                  "molecule is a textbook example of a compound too greasy to be a useful "
                  "drug. It has zero hydrogen bond donors or acceptors — it cannot form the "
                  "stabilising interactions needed to anchor into protein binding sites. "
                  "In biological systems it accumulates non-specifically in lipid membranes. "
                  "NCI activity score: 15/100."
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# 10. Sidebar selector
# ─────────────────────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown("### Experimental Design Matrix")
st.sidebar.markdown("Trial compounds from the NCI dataset are pre-loaded below.")

selected_preset = st.sidebar.selectbox(
    "Select a compound vector:",
    options=list(preset_compounds.keys())
)
chosen_data = preset_compounds[selected_preset]["values"]

# ─────────────────────────────────────────────────────────────────────────────
# 11. Session state sync
# ─────────────────────────────────────────────────────────────────────────────
if "current_preset" not in st.session_state:
    st.session_state.current_preset = selected_preset

if st.session_state.current_preset != selected_preset:
    st.session_state.molwt_val      = chosen_data["MolWt"]
    st.session_state.logp_val       = chosen_data["LogP"]
    st.session_state.donors_val     = chosen_data["NumHDonors"]
    st.session_state.acceptors_val  = chosen_data["NumHAcceptors"]
    st.session_state.tpsa_val       = chosen_data["TPSA"]
    st.session_state.current_preset = selected_preset
    st.session_state.show_scroll_hint = False
    st.session_state.screening_run  = False
    st.session_state.last_result    = None
    st.rerun()

for key in ["molwt_val", "logp_val", "donors_val", "acceptors_val", "tpsa_val"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ─────────────────────────────────────────────────────────────────────────────
# 12. Scroll hint
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.show_scroll_hint:
    st.markdown("""
        <div id="scroll-hint">Scroll down for biochemical insights</div>
        <script>
        (function() {
            var hint = document.getElementById('scroll-hint');
            if (!hint) return;
            window.addEventListener('scroll', function() {
                if (window.scrollY > 60) {
                    hint.style.opacity = '0';
                    setTimeout(function(){ if(hint) hint.style.display='none'; }, 450);
                }
            }, {{ passive: true }});
        })();
        </script>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# 13. Tabs
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["User Control", "User Guide"])

with tab1:
    st.markdown(f"<h3 style='color:{SECTION_HEAD};'>Input Molecular Attributes</h3>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    user_inputs = {}

    with col1:
        user_inputs['MolWt'] = st.number_input(
            "Molecular Weight (MolWt)",
            min_value=0.0, max_value=10000.0,
            value=st.session_state.get("molwt_val", None),
            step=0.001, format="%.3f",
            help="Total molecular mass calculated from atomic contributions."
        )
        user_inputs['LogP'] = st.number_input(
            "Partition Coefficient (LogP)",
            min_value=-20.0, max_value=20.0,
            value=st.session_state.get("logp_val", None),
            step=0.00001, format="%.5f",
            help="Logarithm of the octanol/water partition coefficient."
        )

    with col2:
        user_inputs['NumHDonors'] = st.number_input(
            "H-Bond Donors",
            min_value=0, max_value=50,
            value=st.session_state.get("donors_val", None),
            step=1,
            help="Number of hydrogen atoms bound to electronegative atoms (N, O, F)."
        )
        user_inputs['NumHAcceptors'] = st.number_input(
            "H-Bond Acceptors",
            min_value=0, max_value=50,
            value=st.session_state.get("acceptors_val", None),
            step=1,
            help="Number of electronegative heteroatoms with available lone pairs."
        )

    with col3:
        user_inputs['TPSA'] = st.number_input(
            "Polar Surface Area (TPSA)",
            min_value=0.0, max_value=2000.0,
            value=st.session_state.get("tpsa_val", None),
            step=0.01, format="%.2f",
            help="Topological Polar Surface Area from oxygen, nitrogen, and attached hydrogens."
        )

    st.markdown("---")

    if st.button("Run Molecular Screening", use_container_width=True):
        raw_vals    = [[user_inputs[f] for f in features]]
        pred_proba  = model.predict_proba(raw_vals)[0]
        prob_active = pred_proba[1]
        prediction  = 1 if prob_active >= 0.15 else 0
        st.session_state.last_result = {
            "prob_active":   prob_active,
            "prob_inactive": pred_proba[0],
            "prediction":    prediction,
            "preset":        selected_preset
        }
        st.session_state.screening_run    = True
        st.session_state.show_scroll_hint = True
        st.rerun()

    # ── Results block ─────────────────────────────────────────────────────────
    if st.session_state.screening_run and st.session_state.last_result:
        r            = st.session_state.last_result
        prob_active  = r["prob_active"]
        pred_proba0  = r["prob_inactive"]
        prediction   = r["prediction"]
        result_preset = r["preset"]

        st.markdown(f"<h2 style='color:{SECTION_HEAD}; margin-bottom:4px;'>Screening Analysis</h2>", unsafe_allow_html=True)
        st.markdown(f"<p style='color:{DESC_COLOR}; margin-top:0;'><strong>Compound Class:</strong> {preset_compounds[result_preset]['name']}</p>", unsafe_allow_html=True)

        res_col1, res_col2, res_col3 = st.columns(3)

        with res_col1:
            st.markdown(f"<p style='color:{DESC_COLOR}; margin-bottom:2px; font-size:0.85rem;'>Outcome</p>", unsafe_allow_html=True)
            if prediction == 1:
                st.markdown("<h2 style='color:#ffffff; margin-top:0;'>Active</h2>", unsafe_allow_html=True)
            else:
                st.markdown("<h2 style='color:#ffffff; margin-top:0;'>Inactive</h2>", unsafe_allow_html=True)

        with res_col2:
            st.metric(label="Active Class Probability",   value=f"{prob_active * 100:.1f}%")

        with res_col3:
            st.metric(label="Inactive Class Probability", value=f"{pred_proba0 * 100:.1f}%")

        if prediction == 1:
            st.success("This ligand configuration satisfies structural attributes for targeted bioactivity optimization.")
        else:
            st.warning("In silico screening indicates this physicochemical profile deviates from structural coordinates required for receptor complementarity.")

        st.markdown("---")
        st.markdown(f"<h3 style='color:{SECTION_HEAD};'>Biochemical Insight Note</h3>", unsafe_allow_html=True)
        st.info(preset_compounds[result_preset]['fact'])

with tab2:
    st.markdown(f"<h3 style='color:{SECTION_HEAD};'>Molecular Descriptor Documentation</h3>", unsafe_allow_html=True)

    with st.expander("Lipinski's Rule of 5 — Drug-Likeness Framework"):
        st.write("""
        **Lipinski's Rule of 5** is a set of empirical guidelines developed by Christopher
        Lipinski at Pfizer in 1997 to evaluate whether a chemical compound will be orally
        bioavailable in humans. A compound is considered drug-like if it satisfies most of
        the following criteria:

        * **Molecular Weight (MolWt) <= 500 Da** — Larger molecules struggle to be absorbed
          through the intestinal wall into the bloodstream.

        * **LogP <= 5** — Compounds with very high lipophilicity accumulate in fat tissue and
          are difficult to dissolve in the aqueous environment of the body.

        * **H-Bond Donors <= 5** — Too many donor groups increase polarity, making it harder
          for the molecule to cross lipid-based cell membranes.

        * **H-Bond Acceptors <= 10** — Excess acceptors similarly raise the energetic cost of
          desolvation, reducing membrane permeability.

        A compound that violates more than one of these rules is unlikely to be an effective
        orally administered drug. BioSift evaluates all five descriptors against this framework
        to classify bioactivity potential.
        """)

    with st.expander("Physicochemical Determinants of Pharmacokinetics"):
        st.write("""
        The parameters assessed by this model evaluate drug-likeness based on established
        medicinal chemistry heuristics. These rules predict whether a chemical compound
        possesses molecular properties optimal for oral bioavailability in human clinical settings.
        """)

    with st.expander("Quantitative Lipophilicity (LogP)"):
        st.write("""
        **Octanol-Water Partition Coefficient (LogP):** Measures the thermodynamic equilibrium
        distribution of a solute between an aqueous phase and n-octanol.

        * **Biochemical Impact:** High positive values reflect extreme hydrophobicity, leading
        to non-specific plasma protein binding and metabolic clearance. Negative values indicate
        highly hydrophilic compounds that struggle to cross lipid bilayers passively.
        """)

    with st.expander("Topological Polar Surface Area (TPSA)"):
        st.write("""
        **TPSA:** Quantifies the surface area of polar atoms (nitrogen, oxygen, and attached
        hydrogens) across a stable molecular conformation.

        * **Biochemical Impact:** TPSA is an inverse proxy for passive membrane permeability.
        Compounds exceeding 140 Angstrom squared typically show poor intestinal absorption;
        blood-brain barrier penetrators must generally fall below 90 Angstrom squared.
        """)

    with st.expander("Hydrogen Bonding Capacity (Donors / Acceptors)"):
        st.write("""
        **Hydrogen Bond Kinetics:** Quantifies the structural capacity of a molecule to
        stabilize coordinate links within a target receptor binding site.

        * **Biochemical Impact:** Excessive donor counts (over 5) or acceptor metrics (over 10)
        elevate the free energy barrier for desolvation, disrupting the compound's ability to
        cross cellular barriers efficiently.
        """)