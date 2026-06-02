# ===============================
# STREAMLIT APP - MUGILIDAE FISH CLASSIFIER
# COMPLETE VERSION - 15 FEATURES WITH FULL RANGES
# ===============================

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Mugilidae Fish Classifier", page_icon="🐟", layout="wide")

# ===============================
# COMPLETE SPECIES RANGE DATABASE (15 FEATURES)
# ===============================

SPECIES_RANGES = {
    "Planiliza subviridis": {
        "ND1_Total": (4, 4), "ND2_Total": (6, 9), "NP": (10, 15), "NC": (11, 16),
        "NV_Total": (5, 6), "NA_Total": (8, 11), "SL": (80.44, 622.03),
        "PL": (14.59, 210.91), "BH": (19.57, 227.04), "HL": (21.29, 217.10),
        "Head_Truss": (3.62, 46.87), "Anterior_Truss": (12.81, 47.42),
        "Mid_Truss": (17.49, 87.04), "Posterior_Truss": (17.34, 87.73), "Tail_Truss": (8.55, 67.56)
    },
    "Moolgarda seheli": {
        "ND1_Total": (4, 4), "ND2_Total": (6, 9), "NP": (10, 16), "NC": (11, 17),
        "NV_Total": (5, 7), "NA_Total": (9, 12), "SL": (79.52, 300.00),
        "PL": (21.53, 67.88), "BH": (23.79, 68.95), "HL": (20.79, 75.44),
        "Head_Truss": (3.58, 75.55), "Anterior_Truss": (13.30, 66.92),
        "Mid_Truss": (15.56, 117.48), "Posterior_Truss": (20.44, 127.25), "Tail_Truss": (10.51, 96.42)
    },
    "Osteomugil perusii": {
        "ND1_Total": (4, 4), "ND2_Total": (6, 9), "NP": (10, 16), "NC": (10, 17),
        "NV_Total": (5, 6), "NA_Total": (9, 11), "SL": (11.47, 177.00),
        "PL": (10.86, 160.09), "BH": (12.04, 167.54), "HL": (14.57, 162.09),
        "Head_Truss": (3.63, 76.33), "Anterior_Truss": (10.00, 43.39),
        "Mid_Truss": (10.62, 142.16), "Posterior_Truss": (11.47, 454.04), "Tail_Truss": (6.31, 59.39)
    },
    "Moolgarda tade": {
        "ND1_Total": (4, 4), "ND2_Total": (8, 9), "NP": (15, 17), "NC": (13, 19),
        "NV_Total": (6, 9), "NA_Total": (9, 12), "SL": (74.54, 372.13),
        "PL": (24.25, 75.83), "BH": (30.65, 150.76), "HL": (27.86, 230.02),
        "Head_Truss": (5.09, 87.18), "Anterior_Truss": (14.82, 80.00),
        "Mid_Truss": (24.15, 131.86), "Posterior_Truss": (23.35, 149.08), "Tail_Truss": (13.88, 108.36)
    },
    "Ellochelon vaigiensis": {
        "ND1_Total": (4, 4), "ND2_Total": (6, 9), "NP": (10, 16), "NC": (11, 16),
        "NV_Total": (6, 7), "NA_Total": (7, 11), "SL": (49.73, 363.50),
        "PL": (0, 53.86), "BH": (30.5, 119.05), "HL": (31.61, 183.42),
        "Head_Truss": (4.99, 92.38), "Anterior_Truss": (18.22, 125.79),
        "Mid_Truss": (18.30, 148.39), "Posterior_Truss": (27.28, 168.88), "Tail_Truss": (15.73, 115.95)
    }
}

# Feature names in order
FEATURE_NAMES = [
    "ND1_Total", "ND2_Total", "NP", "NC", "NV_Total", "NA_Total",
    "SL", "PL", "BH", "HL", "Head_Truss", "Anterior_Truss",
    "Mid_Truss", "Posterior_Truss", "Tail_Truss"
]

# Feature display names for UI
FEATURE_DISPLAY = [
    "ND1_Total", "ND2_Total", "NP", "NC", "NV_Total", "NA_Total",
    "SL (mm)", "PL (mm)", "BH (mm)", "HL (mm)", "Head_Truss (mm)",
    "Anterior_Truss (mm)", "Mid_Truss (mm)", "Posterior_Truss (mm)", "Tail_Truss (mm)"
]

def predict_by_range(features):
    """Predict species based on measurement ranges"""
    scores = {}
    for species, ranges in SPECIES_RANGES.items():
        matches = 0
        for i, f in enumerate(FEATURE_NAMES):
            mn, mx = ranges[f]
            if mn <= features[i] <= mx:
                matches += 1
        scores[species] = (matches / len(FEATURE_NAMES)) * 100
    
    best_species = max(scores, key=scores.get)
    return best_species, scores

# ===============================
# INITIALIZE SESSION STATE
# ===============================

if 'features' not in st.session_state:
    st.session_state.features = {
        "ND1_Total": 4.0, "ND2_Total": 7.0, "NP": 14.0, "NC": 14.0,
        "NV_Total": 6.0, "NA_Total": 10.0, "SL": 150.0, "PL": 40.0,
        "BH": 45.0, "HL": 40.0, "Head_Truss": 50.0, "Anterior_Truss": 40.0,
        "Mid_Truss": 80.0, "Posterior_Truss": 80.0, "Tail_Truss": 50.0
    }

# ===============================
# SIDEBAR
# ===============================

st.sidebar.title("🐟 Mugilidae Fish Classifier")
st.sidebar.markdown("---")
st.sidebar.header("About")
st.sidebar.info("""
**5 Mugilidae Species:**
- Planiliza subviridis
- Moolgarda seheli
- Osteomugil perusii
- Moolgarda tade
- Ellochelon vaigiensis

**15 Features:**
- Meristic: ND1, ND2, NP, NC, NV, NA
- Morphometric: SL, PL, BH, HL
- Truss: Head, Anterior, Mid, Posterior, Tail

**Method:** Range-based matching
""")
st.sidebar.caption("FYP Project | UMT")

# ===============================
# MAIN TITLE
# ===============================

st.title("🐟 Mugilidae Fish Classification System")
st.markdown("### Identify Mullet Species Using 15 Morphometric Measurements")
st.markdown("---")

# ===============================
# SPECIES REFERENCE TABLE
# ===============================

st.header("📖 Species Measurement Ranges")

range_data = []
for sp, ranges in SPECIES_RANGES.items():
    range_data.append({
        "Species": sp,
        "ND2": f"{ranges['ND2_Total'][0]}-{ranges['ND2_Total'][1]}",
        "NP": f"{ranges['NP'][0]}-{ranges['NP'][1]}",
        "NC": f"{ranges['NC'][0]}-{ranges['NC'][1]}",
        "SL (mm)": f"{ranges['SL'][0]:.0f}-{ranges['SL'][1]:.0f}",
        "Size": "Large" if ranges['SL'][1] > 250 else "Small-Medium"
    })

st.dataframe(pd.DataFrame(range_data), use_container_width=True)

st.info("""
💡 **Key Differentiators:**
- **Moolgarda tade**: NP (15-17) and ND2 (8-9) are HIGHER
- **Large species** (SL > 250mm): Planiliza subviridis, Moolgarda tade
- **Small-Medium species** (SL < 200mm): Moolgarda seheli, Osteomugil perusii, Ellochelon vaigiensis
""")

# ===============================
# QUICK LOAD BUTTONS
# ===============================

st.subheader("Quick Load - Sample Values")

col1, col2, col3, col4, col5 = st.columns(5)

# Sample values for each species
sample_values = {
    "Planiliza subviridis": [4, 7, 13, 14, 6, 10, 300, 50, 60, 55, 25, 30, 50, 50, 35],
    "Moolgarda seheli": [4, 7, 14, 15, 6, 10, 150, 35, 40, 35, 30, 35, 60, 65, 45],
    "Osteomugil perusii": [4, 7, 13, 14, 6, 10, 140, 35, 40, 35, 30, 30, 55, 60, 40],
    "Moolgarda tade": [4, 8, 16, 16, 7, 10, 300, 45, 55, 50, 40, 45, 70, 75, 55],
    "Ellochelon vaigiensis": [4, 7, 13, 14, 6, 10, 150, 30, 45, 40, 30, 40, 60, 70, 50]
}

def set_sample_values(species):
    vals = sample_values[species]
    st.session_state.features = {
        "ND1_Total": vals[0], "ND2_Total": vals[1], "NP": vals[2], "NC": vals[3],
        "NV_Total": vals[4], "NA_Total": vals[5], "SL": vals[6], "PL": vals[7],
        "BH": vals[8], "HL": vals[9], "Head_Truss": vals[10], "Anterior_Truss": vals[11],
        "Mid_Truss": vals[12], "Posterior_Truss": vals[13], "Tail_Truss": vals[14]
    }

if col1.button("📌 Planiliza", key="btn_planiliza"):
    set_sample_values("Planiliza subviridis")
if col2.button("📌 Moolgarda s", key="btn_moolgarda_s"):
    set_sample_values("Moolgarda seheli")
if col3.button("📌 Osteomugil", key="btn_osteomugil"):
    set_sample_values("Osteomugil perusii")
if col4.button("📌 Moolgarda t", key="btn_moolgarda_t"):
    set_sample_values("Moolgarda tade")
if col5.button("📌 Ellochelon", key="btn_ellochelon"):
    set_sample_values("Ellochelon vaigiensis")

st.markdown("---")

# ===============================
# INPUT FORM - 15 FEATURES
# ===============================

st.header("🔮 Enter Fish Measurements")

# Create 3 columns for better layout
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Meristic Features")
    nd1 = st.number_input("ND1_Total", value=st.session_state.features["ND1_Total"], step=1.0, key="inp_nd1")
    nd2 = st.number_input("ND2_Total", value=st.session_state.features["ND2_Total"], step=1.0, key="inp_nd2")
    np_val = st.number_input("NP", value=st.session_state.features["NP"], step=1.0, key="inp_np")
    nc = st.number_input("NC", value=st.session_state.features["NC"], step=1.0, key="inp_nc")
    nv = st.number_input("NV_Total", value=st.session_state.features["NV_Total"], step=1.0, key="inp_nv")
    na = st.number_input("NA_Total", value=st.session_state.features["NA_Total"], step=1.0, key="inp_na")

with col2:
    st.subheader("Morphometric Features (mm)")
    sl = st.number_input("SL", value=st.session_state.features["SL"], step=10.0, key="inp_sl")
    pl = st.number_input("PL", value=st.session_state.features["PL"], step=5.0, key="inp_pl")
    bh = st.number_input("BH", value=st.session_state.features["BH"], step=5.0, key="inp_bh")
    hl = st.number_input("HL", value=st.session_state.features["HL"], step=5.0, key="inp_hl")

with col3:
    st.subheader("Truss Features (mm)")
    head = st.number_input("Head_Truss", value=st.session_state.features["Head_Truss"], step=5.0, key="inp_head")
    ant = st.number_input("Anterior_Truss", value=st.session_state.features["Anterior_Truss"], step=5.0, key="inp_ant")
    mid = st.number_input("Mid_Truss", value=st.session_state.features["Mid_Truss"], step=10.0, key="inp_mid")
    post = st.number_input("Posterior_Truss", value=st.session_state.features["Posterior_Truss"], step=10.0, key="inp_post")
    tail = st.number_input("Tail_Truss", value=st.session_state.features["Tail_Truss"], step=5.0, key="inp_tail")

# Update session state
st.session_state.features = {
    "ND1_Total": nd1, "ND2_Total": nd2, "NP": np_val, "NC": nc,
    "NV_Total": nv, "NA_Total": na, "SL": sl, "PL": pl,
    "BH": bh, "HL": hl, "Head_Truss": head, "Anterior_Truss": ant,
    "Mid_Truss": mid, "Posterior_Truss": post, "Tail_Truss": tail
}

# ===============================
# PREDICT BUTTON
# ===============================

if st.button("🔍 Identify Species", type="primary", key="btn_predict"):
    features = [nd1, nd2, np_val, nc, nv, na, sl, pl, bh, hl, head, ant, mid, post, tail]
    
    # Predict using range matching
    predicted, scores = predict_by_range(features)
    
    # Display result
    st.markdown("---")
    st.success(f"### 🎯 Predicted Species: **{predicted}**")
    
    # Show confidence
    confidence = scores[predicted]
    st.progress(int(confidence))
    st.caption(f"Measurement compatibility: {confidence:.1f}%")
    
    # Show all species scores
    st.subheader("Compatibility with all species")
    
    score_df = pd.DataFrame({
        'Species': list(scores.keys()),
        'Compatibility (%)': list(scores.values())
    }).sort_values('Compatibility (%)', ascending=False)
    
    # Add color coding
    def color_compatibility(val):
        if val >= 70:
            return 'background-color: #90EE90'
        elif val >= 50:
            return 'background-color: #FFD700'
        else:
            return 'background-color: #FFB6C1'
    
    st.dataframe(score_df.style.applymap(color_compatibility, subset=['Compatibility (%)']), use_container_width=True)
    
    # Show detailed feature analysis
    st.subheader("Feature Analysis")
    
    ranges = SPECIES_RANGES[predicted]
    analysis_data = []
    
    for i, f in enumerate(FEATURE_NAMES):
        mn, mx = ranges[f]
        val = features[i]
        status = "✅" if mn <= val <= mx else "❌"
        analysis_data.append({
            "Feature": FEATURE_DISPLAY[i],
            "Your Value": f"{val:.1f}",
            "Typical Range": f"{mn:.1f} - {mx:.1f}",
            "Status": status
        })
    
    st.dataframe(pd.DataFrame(analysis_data), use_container_width=True)
    
    # Warning if low confidence
    if confidence < 50:
        st.warning("⚠️ **Low confidence** - The measurements are quite different from typical values. Please double-check the measurements or refer to the species ranges table above.")
    elif confidence >= 70:
        st.success("✅ **High confidence** - Measurements are consistent with typical values for this species!")
    
    # Size-based suggestion
    if sl > 250:
        st.info("📏 **Note:** SL > 250mm indicates a LARGE species. Possible species: Planiliza subviridis or Moolgarda tade")
    elif sl < 200:
        st.info("📏 **Note:** SL < 200mm indicates a SMALL-MEDIUM species. Possible species: Moolgarda seheli, Osteomugil perusii, or Ellochelon vaigiensis")
    
    # ND2 and NP suggestion
    if nd2 >= 8 and np_val >= 15:
        st.info("🔍 **Note:** High ND2 (≥8) and NP (≥15) strongly suggests Moolgarda tade")

# ===============================
# EXPANDED REFERENCE TABLE (ALL FEATURES)
# ===============================

with st.expander("📖 Complete Species Ranges (All 15 Features)"):
    all_ranges_data = []
    for sp, ranges in SPECIES_RANGES.items():
        row = {"Species": sp}
        for f in FEATURE_NAMES:
            mn, mx = ranges[f]
            row[f] = f"{mn:.1f}-{mx:.1f}"
        all_ranges_data.append(row)
    
    st.dataframe(pd.DataFrame(all_ranges_data), use_container_width=True)

# ===============================
# HOW TO USE
# ===============================

with st.expander("📖 How to Use This App"):
    st.markdown("""
    ### Step-by-Step Guide:
    
    1. **Refer to the species ranges table** above to understand typical measurements for each species
    
    2. **Enter your fish measurements** in the input fields (15 features)
    
    3. **Click 'Identify Species'** to get prediction
    
    4. **Check the compatibility score** - higher percentage means better match
    
    5. **Review feature analysis** to see which measurements match or deviate from typical ranges
    
    ### Tips for accurate identification:
    - **ND2_Total and NP** are key differentiators for Moolgarda tade (higher values)
    - **SL (Standard Length)** helps distinguish large vs small species
    - **A compatibility score above 70%** indicates good match
    - Use the **Quick Load buttons** to test with sample values
    
    ### Features Description:
    - **Meristic**: Fin ray/spine counts (ND1, ND2, NP, NC, NV, NA)
    - **Morphometric**: Body measurements (SL, PL, BH, HL)
    - **Truss**: Distance between anatomical landmarks
    """)

# ===============================
# FOOTER
# ===============================

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
    <p>🐟 Mugilidae Fish Classification using 15 Morphometric Measurements</p>
    <p>FYP Project | Universiti Malaysia Terengganu</p>
    </div>
    """,
    unsafe_allow_html=True
)
