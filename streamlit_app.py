# ===============================
# STREAMLIT APP - MUGILIDAE FISH CLASSIFIER
# WITH FULL EXTRACTED DATA (15 FEATURES INCLUDING TRUSS SUMS)
# ===============================

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Mugilidae Fish Classifier", page_icon="🐟", layout="wide")

# ===============================
# FUNCTIONS
# ===============================

def extract_block(df, keyword):
    """Extract data block from Excel sheet"""
    first_col = df.iloc[:, 0].astype(str).str.strip().str.lower()
    matches = first_col[first_col == keyword.lower()].index
    
    if len(matches) == 0:
        return None
    
    start_idx = matches[0]
    header_row = start_idx + 1
    data_start = start_idx + 2
    
    # Get headers
    headers = []
    for h in df.iloc[header_row]:
        if pd.notna(h) and str(h).strip() != '':
            headers.append(str(h).strip())
    
    # Extract data rows
    data = []
    i = data_start
    while i < len(df):
        row = df.iloc[i]
        if pd.isna(row[0]) or str(row[0]).strip() == '':
            break
        
        numeric_row = []
        for val in row[:len(headers)]:
            try:
                numeric_row.append(float(val) if pd.notna(val) else np.nan)
            except (ValueError, TypeError):
                numeric_row.append(np.nan)
        data.append(numeric_row)
        i += 1
    
    if not data:
        return None
    
    df_block = pd.DataFrame(data, columns=headers[:len(data[0])])
    
    # Remove Specimen column
    if 'Specimen' in df_block.columns:
        df_block = df_block.drop('Specimen', axis=1)
    
    return df_block

def get_truss_sum(truss_df, measurements):
    """Sum specific truss measurements"""
    truss_cols = {str(col).replace(' ', '').replace('-', ''): col for col in truss_df.columns}
    total = np.zeros(len(truss_df))
    
    for meas in measurements:
        meas_clean = meas.replace('-', '')
        for key, col in truss_cols.items():
            if meas_clean == key or meas_clean in key or key in meas_clean:
                total += truss_df[col].fillna(0).values
                break
    return total

def process_species_data(excel_file, sheet_idx, species_name):
    """Extract all 15 features for a species"""
    df_raw = pd.read_excel(excel_file, sheet_name=sheet_idx, header=None)
    
    # Extract blocks
    meristic = extract_block(df_raw, "Meristic")
    morphometric = extract_block(df_raw, "Morphometric")
    truss = extract_block(df_raw, "Truss Network")
    if truss is None:
        truss = extract_block(df_raw, "Truss")
    
    if meristic is None or morphometric is None or truss is None:
        return None
    
    n = min(len(meristic), len(morphometric), len(truss))
    meristic = meristic.iloc[:n].reset_index(drop=True)
    morphometric = morphometric.iloc[:n].reset_index(drop=True)
    truss = truss.iloc[:n].reset_index(drop=True)
    
    # MERISTIC FEATURES
    nd1_cols = [col for col in meristic.columns if 'ND1' in str(col)]
    nd1_total = meristic[nd1_cols].sum(axis=1).fillna(0).values if nd1_cols else np.zeros(n)
    
    nd2_cols = [col for col in meristic.columns if 'ND2' in str(col)]
    nd2_total = meristic[nd2_cols].sum(axis=1).fillna(0).values if nd2_cols else np.zeros(n)
    
    np_val = meristic['NP'].fillna(0).values if 'NP' in meristic.columns else np.zeros(n)
    nc_val = meristic['NC'].fillna(0).values if 'NC' in meristic.columns else np.zeros(n)
    
    nv_cols = [col for col in meristic.columns if 'NV' in str(col)]
    nv_total = meristic[nv_cols].sum(axis=1).fillna(0).values if nv_cols else np.zeros(n)
    
    na_cols = [col for col in meristic.columns if 'NA' in str(col)]
    na_total = meristic[na_cols].sum(axis=1).fillna(0).values if na_cols else np.zeros(n)
    
    # MORPHOMETRIC FEATURES
    sl = morphometric['SL'].fillna(0).values if 'SL' in morphometric.columns else np.zeros(n)
    pl = morphometric['PL'].fillna(0).values if 'PL' in morphometric.columns else np.zeros(n)
    bh = morphometric['BH'].fillna(0).values if 'BH' in morphometric.columns else np.zeros(n)
    hl = morphometric['HL'].fillna(0).values if 'HL' in morphometric.columns else np.zeros(n)
    
    # TRUSS FEATURES (SUMS)
    head_truss = get_truss_sum(truss, ['AB', 'AC', 'AD'])
    anterior_truss = get_truss_sum(truss, ['BC', 'BD', 'CD'])
    mid_truss = get_truss_sum(truss, ['CE', 'CF', 'DE', 'DF', 'EF'])
    posterior_truss = get_truss_sum(truss, ['EG', 'EH', 'FG', 'FH', 'GH'])
    tail_truss = get_truss_sum(truss, ['GI', 'GJ', 'HI', 'HJ', 'IJ'])
    
    # Create DataFrame
    species_df = pd.DataFrame({
        'Species': [species_name] * n,
        'ND1_Total': nd1_total,
        'ND2_Total': nd2_total,
        'NP': np_val,
        'NC': nc_val,
        'NV_Total': nv_total,
        'NA_Total': na_total,
        'SL': sl,
        'PL': pl,
        'BH': bh,
        'HL': hl,
        'Head_Truss': head_truss,
        'Anterior_Truss': anterior_truss,
        'Mid_Truss': mid_truss,
        'Posterior_Truss': posterior_truss,
        'Tail_Truss': tail_truss
    })
    
    return species_df

# ===============================
# FEATURE NAMES
# ===============================

FEATURE_NAMES = [
    "ND1_Total", "ND2_Total", "NP", "NC", "NV_Total", "NA_Total",
    "SL", "PL", "BH", "HL", "Head_Truss", "Anterior_Truss",
    "Mid_Truss", "Posterior_Truss", "Tail_Truss"
]

FEATURE_DISPLAY = [
    "ND1_Total", "ND2_Total", "NP", "NC", "NV_Total", "NA_Total",
    "SL (mm)", "PL (mm)", "BH (mm)", "HL (mm)",
    "Head_Truss (mm)", "Anterior_Truss (mm)", "Mid_Truss (mm)",
    "Posterior_Truss (mm)", "Tail_Truss (mm)"
]

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
- Meristic (6): ND1, ND2, NP, NC, NV, NA
- Morphometric (4): SL, PL, BH, HL
- Truss (5): Head, Anterior, Mid, Posterior, Tail
""")
st.sidebar.caption("FYP Project | UMT")

# ===============================
# MAIN TITLE
# ===============================

st.title("🐟 Mugilidae Fish Classification System")
st.markdown("### Identify Mullet Species Using 15 Morphometric Measurements")
st.markdown("---")

# ===============================
# FILE UPLOAD
# ===============================

st.header("Step 1: Upload Your Excel File")

uploaded_file = st.file_uploader(
    "Upload FYP Mugilidae Dataset(CLEANED).xlsx",
    type=['xlsx'],
    help="Upload your Excel file containing meristic, morphometric, and truss measurements"
)

if uploaded_file is not None:
    
    # ===============================
    # PROCESS ALL SPECIES
    # ===============================
    
    with st.spinner("Extracting data from Excel..."):
        
        species_names = [
            "Planiliza subviridis",
            "Moolgarda seheli",
            "Osteomugil perusii",
            "Moolgarda tade",
            "Ellochelon vaigiensis"
        ]
        
        all_data = []
        for idx, species in enumerate(species_names):
            df_species = process_species_data(uploaded_file, idx, species)
            if df_species is not None:
                all_data.append(df_species)
                st.success(f"✓ Loaded {len(df_species)} specimens for {species}")
        
        if not all_data:
            st.error("Failed to load data. Please check your Excel file format.")
            st.stop()
        
        df = pd.concat(all_data, ignore_index=True)
        
        # Clean data
        for col in FEATURE_NAMES:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(df[col].median())
        
        # Calculate ranges for each species
        SPECIES_RANGES = {}
        for species in species_names:
            species_data = df[df['Species'] == species]
            SPECIES_RANGES[species] = {}
            for f in FEATURE_NAMES:
                SPECIES_RANGES[species][f] = (
                    species_data[f].min(),
                    species_data[f].max()
                )
        
        # Calculate sample values (mean of each feature per species)
        SAMPLE_VALUES = {}
        for species in species_names:
            species_data = df[df['Species'] == species]
            SAMPLE_VALUES[species] = [
                species_data[f].mean() for f in FEATURE_NAMES
            ]
    
    st.success(f"✅ Data loaded! Total: {len(df)} specimens across 5 species")
    
    with st.expander("Preview Data"):
        st.dataframe(df.head(10))
    
    # ===============================
    # SPECIES REFERENCE TABLE
    # ===============================
    
    st.header("📖 Species Measurement Ranges (from your data)")
    
    range_data = []
    for sp in species_names:
        ranges = SPECIES_RANGES[sp]
        range_data.append({
            "Species": sp,
            "ND2": f"{ranges['ND2_Total'][0]:.0f}-{ranges['ND2_Total'][1]:.0f}",
            "NP": f"{ranges['NP'][0]:.0f}-{ranges['NP'][1]:.0f}",
            "SL": f"{ranges['SL'][0]:.0f}-{ranges['SL'][1]:.0f} mm",
            "Head_Truss": f"{ranges['Head_Truss'][0]:.0f}-{ranges['Head_Truss'][1]:.0f}"
        })
    
    st.dataframe(pd.DataFrame(range_data), use_container_width=True)
    
    st.info("""
    💡 **Key Differentiators:**
    - **Moolgarda tade**: NP (15-17) and ND2 (8-9) are HIGHER
    - **Large species** (SL > 250mm): Planiliza subviridis, Moolgarda tade
    """)
    
    # ===============================
    # QUICK LOAD BUTTONS
    # ===============================
    
    st.subheader("Quick Load - Sample Values (Mean from your data)")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    def set_sample_values(species):
        vals = SAMPLE_VALUES[species]
        st.session_state['sample_vals'] = vals
        st.session_state['selected_species'] = species
    
    for i, sp in enumerate(species_names):
        btn_label = sp.split()[0]
        if i == 0 and col1.button(f"📌 {btn_label}", key="btn_planiliza"):
            set_sample_values(sp)
        elif i == 1 and col2.button(f"📌 {btn_label}", key="btn_moolgarda_s"):
            set_sample_values(sp)
        elif i == 2 and col3.button(f"📌 {btn_label}", key="btn_osteomugil"):
            set_sample_values(sp)
        elif i == 3 and col4.button(f"📌 {btn_label}", key="btn_moolgarda_t"):
            set_sample_values(sp)
        elif i == 4 and col5.button(f"📌 {btn_label}", key="btn_ellochelon"):
            set_sample_values(sp)
    
    st.markdown("---")
    
    # ===============================
    # INPUT FORM (15 FEATURES)
    # ===============================
    
    st.header("🔮 Enter Fish Measurements")
    
    # Default values
    if 'sample_vals' not in st.session_state:
        default_vals = [4, 7, 14, 14, 6, 10, 150, 40, 45, 40, 50, 45, 80, 80, 50]
    else:
        default_vals = st.session_state['sample_vals']
    
    # Display selected species if any
    if 'selected_species' in st.session_state:
        st.info(f"📌 Loaded sample values for: **{st.session_state['selected_species']}**")
    
    # Create columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Meristic Features")
        nd1 = st.number_input("ND1_Total", value=float(default_vals[0]), step=1.0, key="inp_nd1")
        nd2 = st.number_input("ND2_Total", value=float(default_vals[1]), step=1.0, key="inp_nd2")
        np_val = st.number_input("NP", value=float(default_vals[2]), step=1.0, key="inp_np")
        nc = st.number_input("NC", value=float(default_vals[3]), step=1.0, key="inp_nc")
        nv = st.number_input("NV_Total", value=float(default_vals[4]), step=1.0, key="inp_nv")
        na = st.number_input("NA_Total", value=float(default_vals[5]), step=1.0, key="inp_na")
    
    with col2:
        st.subheader("Morphometric Features (mm)")
        sl = st.number_input("SL", value=float(default_vals[6]), step=10.0, key="inp_sl")
        pl = st.number_input("PL", value=float(default_vals[7]), step=5.0, key="inp_pl")
        bh = st.number_input("BH", value=float(default_vals[8]), step=5.0, key="inp_bh")
        hl = st.number_input("HL", value=float(default_vals[9]), step=5.0, key="inp_hl")
    
    with col3:
        st.subheader("Truss Features (mm)")
        head = st.number_input("Head_Truss", value=float(default_vals[10]), step=10.0, key="inp_head")
        ant = st.number_input("Anterior_Truss", value=float(default_vals[11]), step=10.0, key="inp_ant")
        mid = st.number_input("Mid_Truss", value=float(default_vals[12]), step=20.0, key="inp_mid")
        post = st.number_input("Posterior_Truss", value=float(default_vals[13]), step=20.0, key="inp_post")
        tail = st.number_input("Tail_Truss", value=float(default_vals[14]), step=10.0, key="inp_tail")
    
    features = [nd1, nd2, np_val, nc, nv, na, sl, pl, bh, hl, head, ant, mid, post, tail]
    
    # ===============================
    # PREDICTION FUNCTION
    # ===============================
    
    def predict_by_range(features):
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
    # PREDICT BUTTON
    # ===============================
    
    if st.button("🔍 Identify Species", type="primary", key="btn_predict"):
        predicted, scores = predict_by_range(features)
        
        st.markdown("---")
        st.success(f"### 🎯 Predicted Species: **{predicted}**")
        
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
        
        # Detailed feature analysis
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
        
        # Warnings and tips
        if confidence < 50:
            st.warning("⚠️ Low confidence - measurements may be outside typical ranges.")
        elif confidence >= 70:
            st.success("✅ High confidence - measurements are consistent with typical values!")
        
        if sl > 250:
            st.info("📏 Note: SL > 250mm indicates a LARGE species")
        elif sl < 200:
            st.info("📏 Note: SL < 200mm indicates a SMALL-MEDIUM species")
        
        if nd2 >= 8 and np_val >= 15:
            st.info("🔍 Note: High ND2 (≥8) and NP (≥15) strongly suggests Moolgarda tade")

else:
    st.info("👈 Please upload your Excel file to begin")
    
    with st.expander("📖 How to Use This App"):
        st.markdown("""
        1. **Upload** your Excel file (FYP Mugilidae Dataset(CLEANED).xlsx)
        2. The system will automatically extract all 15 features
        3. **Enter measurements** or use Quick Load buttons
        4. **Click Identify Species** to see prediction
        5. **Review compatibility** and feature analysis
        """)

# ===============================
# FOOTER
# ===============================

st.markdown("---")
st.caption("FYP Project | Universiti Malaysia Terengganu")
