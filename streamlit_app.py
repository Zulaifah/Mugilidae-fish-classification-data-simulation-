# ===============================
# STREAMLIT APP - MUGILIDAE FISH CLASSIFIER
# SIMPLIFIED VERSION - PREDICTION ONLY
# ===============================

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Mugilidae Fish Classifier", page_icon="🐟", layout="wide")

# ===============================
# SPECIES RANGE DATABASE
# ===============================

SPECIES_RANGES = {
    "Planiliza subviridis": {
        "ND1_Total": (4, 4), "ND2_Total": (6, 9), "NP": (10, 15), "NC": (11, 16),
        "NV_Total": (5, 6), "NA_Total": (8, 11), "SL": (80, 622)
    },
    "Moolgarda seheli": {
        "ND1_Total": (4, 4), "ND2_Total": (6, 9), "NP": (10, 16), "NC": (11, 17),
        "NV_Total": (5, 7), "NA_Total": (9, 12), "SL": (80, 300)
    },
    "Osteomugil perusii": {
        "ND1_Total": (4, 4), "ND2_Total": (6, 9), "NP": (10, 16), "NC": (10, 17),
        "NV_Total": (5, 6), "NA_Total": (9, 11), "SL": (11, 177)
    },
    "Moolgarda tade": {
        "ND1_Total": (4, 4), "ND2_Total": (8, 9), "NP": (15, 17), "NC": (13, 19),
        "NV_Total": (6, 9), "NA_Total": (9, 12), "SL": (75, 372)
    },
    "Ellochelon vaigiensis": {
        "ND1_Total": (4, 4), "ND2_Total": (6, 9), "NP": (10, 16), "NC": (11, 16),
        "NV_Total": (6, 7), "NA_Total": (7, 11), "SL": (50, 364)
    }
}

FEATURE_NAMES = ["ND1_Total", "ND2_Total", "NP", "NC", "NV_Total", "NA_Total", "SL"]

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

**Features used:**
- ND1_Total, ND2_Total, NP, NC
- NV_Total, NA_Total, SL

**Method:** Range-based matching
""")
st.sidebar.caption("FYP Project | UMT")

# ===============================
# MAIN TITLE
# ===============================

st.title("🐟 Mugilidae Fish Classification System")
st.markdown("### Identify Mullet Species Using Measurement Ranges")
st.markdown("---")

# ===============================
# SPECIES REFERENCE TABLE
# ===============================

st.header("📖 Species Measurement Ranges")

range_data = []
for sp, ranges in SPECIES_RANGES.items():
    range_data.append({
        "Species": sp,
        "ND2_Total": f"{ranges['ND2_Total'][0]}-{ranges['ND2_Total'][1]}",
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
# PREDICTION SECTION
# ===============================

st.header("🔮 Identify Fish Species")

st.markdown("### Enter the measurements below")

# Quick reference buttons
st.subheader("Quick Load - Sample Values")
cols = st.columns(5)

sample_values = {
    "Planiliza subviridis": [4, 7, 13, 14, 6, 10, 300],
    "Moolgarda seheli": [4, 7, 14, 15, 6, 10, 150],
    "Osteomugil perusii": [4, 7, 13, 14, 6, 10, 140],
    "Moolgarda tade": [4, 8, 16, 16, 7, 10, 300],
    "Ellochelon vaigiensis": [4, 7, 13, 14, 6, 10, 150]
}

def set_values(species):
    vals = sample_values[species]
    st.session_state['nd1'] = vals[0]
    st.session_state['nd2'] = vals[1]
    st.session_state['np'] = vals[2]
    st.session_state['nc'] = vals[3]
    st.session_state['nv'] = vals[4]
    st.session_state['na'] = vals[5]
    st.session_state['sl'] = vals[6]

for i, sp in enumerate(SPECIES_RANGES.keys()):
    if cols[i].button(f"📌 {sp.split()[0]}"):
        set_values(sp)

# Initialize session state
if 'nd1' not in st.session_state:
    st.session_state['nd1'] = 4
    st.session_state['nd2'] = 7
    st.session_state['np'] = 14
    st.session_state['nc'] = 14
    st.session_state['nv'] = 6
    st.session_state['na'] = 10
    st.session_state['sl'] = 150

# Input form
col1, col2 = st.columns(2)

with col1:
    st.subheader("Meristic Features")
    nd1 = st.number_input("ND1_Total", value=float(st.session_state['nd1']), step=1.0, key="input_nd1")
    nd2 = st.number_input("ND2_Total", value=float(st.session_state['nd2']), step=1.0, key="input_nd2")
    np_val = st.number_input("NP", value=float(st.session_state['np']), step=1.0, key="input_np")
    nc = st.number_input("NC", value=float(st.session_state['nc']), step=1.0, key="input_nc")

with col2:
    st.subheader("Other Features")
    nv = st.number_input("NV_Total", value=float(st.session_state['nv']), step=1.0, key="input_nv")
    na = st.number_input("NA_Total", value=float(st.session_state['na']), step=1.0, key="input_na")
    sl = st.number_input("SL (mm)", value=float(st.session_state['sl']), step=10.0, key="input_sl")

# Predict button
if st.button("🔍 Identify Species", type="primary"):
    features = [nd1, nd2, np_val, nc, nv, na, sl]
    
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
    
    st.dataframe(score_df, use_container_width=True)
    
    # Show which features are out of range
    st.subheader("Feature Analysis")
    
    ranges = SPECIES_RANGES[predicted]
    analysis_data = []
    feature_display = ["ND1_Total", "ND2_Total", "NP", "NC", "NV_Total", "NA_Total", "SL (mm)"]
    values = [nd1, nd2, np_val, nc, nv, na, sl]
    
    for i, f in enumerate(FEATURE_NAMES):
        mn, mx = ranges[f]
        val = values[i]
        status = "✅" if mn <= val <= mx else "❌"
        analysis_data.append({
            "Feature": feature_display[i],
            "Your Value": f"{val:.1f}",
            "Range": f"{mn:.0f}-{mx:.0f}",
            "Status": status
        })
    
    st.dataframe(pd.DataFrame(analysis_data), use_container_width=True)
    
    # Warning if low confidence
    if confidence < 50:
        st.warning("⚠️ **Low confidence** - The measurements are quite different from typical values. Please double-check the measurements or refer to the species ranges table above.")
    
    # Size-based suggestion
    if sl > 250:
        st.info("📏 **Note:** SL > 250mm indicates a LARGE species. Possible species: Planiliza subviridis or Moolgarda tade")
    elif sl < 200:
        st.info("📏 **Note:** SL < 200mm indicates a SMALL-MEDIUM species. Possible species: Moolgarda seheli, Osteomugil perusii, or Ellochelon vaigiensis")

# ===============================
# HOW TO USE
# ===============================

with st.expander("📖 How to Use This App"):
    st.markdown("""
    1. **Refer to the species ranges table** above to understand typical measurements for each species
    
    2. **Enter your fish measurements** in the input fields
    
    3. **Click 'Identify Species'** to get prediction
    
    4. **Check the compatibility score** - higher percentage means better match
    
    5. **Review feature analysis** to see which measurements match or deviate from typical ranges
    
    **Tips for accurate identification:**
    - ND2_Total and NP are key differentiators for Moolgarda tade
    - SL (Standard Length) helps distinguish large vs small species
    - A compatibility score above 70% indicates good match
    """)

# ===============================
# FOOTER
# ===============================

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
    <p>🐟 Mugilidae Fish Classification using Morphometric Ranges</p>
    <p>FYP Project | Universiti Malaysia Terengganu</p>
    </div>
    """,
    unsafe_allow_html=True
)
