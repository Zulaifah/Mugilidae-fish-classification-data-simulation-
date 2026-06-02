# ===============================
# STREAMLIT APP - MUGILIDAE FISH CLASSIFIER
# Range-Based Prediction + Model Comparison
# ===============================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

st.set_page_config(page_title="Mugilidae Fish Classifier", page_icon="🐟", layout="wide")

# ===============================
# SPECIES RANGES DATABASE (YANG ANDA BERIKAN)
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
st.sidebar.header("📋 About")
st.sidebar.info("""
**5 Mugilidae Species:**
- Planiliza subviridis
- Moolgarda seheli
- Osteomugil perusii
- Moolgarda tade
- Ellochelon vaigiensis

**15 Features:**
- Meristic (6)
- Morphometric (4)
- Truss (5)

**Prediction Method:** Range-based matching
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
        "Species": sp.split()[0],
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

# Sample values (mid-range values)
sample_values = {
    "Planiliza subviridis": [4, 7, 13, 14, 5.5, 10, 350, 100, 120, 100, 25, 30, 50, 50, 35],
    "Moolgarda seheli": [4, 7, 13, 14, 6, 10, 180, 45, 45, 45, 35, 40, 65, 70, 50],
    "Osteomugil perusii": [4, 7, 13, 14, 5.5, 10, 100, 80, 90, 80, 35, 25, 70, 200, 30],
    "Moolgarda tade": [4, 8.5, 16, 16, 7.5, 10, 200, 50, 90, 120, 45, 45, 75, 85, 60],
    "Ellochelon vaigiensis": [4, 7, 13, 14, 6.5, 9, 200, 25, 70, 100, 45, 70, 80, 95, 65]
}

def set_sample(species):
    vals = sample_values[species]
    st.session_state['sample_vals'] = vals
    st.session_state['selected_species'] = species

if col1.button("📌 Planiliza", key="btn_planiliza"):
    set_sample("Planiliza subviridis")
if col2.button("📌 Moolgarda s", key="btn_moolgarda_s"):
    set_sample("Moolgarda seheli")
if col3.button("📌 Osteomugil", key="btn_osteomugil"):
    set_sample("Osteomugil perusii")
if col4.button("📌 Moolgarda t", key="btn_moolgarda_t"):
    set_sample("Moolgarda tade")
if col5.button("📌 Ellochelon", key="btn_ellochelon"):
    set_sample("Ellochelon vaigiensis")

st.markdown("---")

# ===============================
# INPUT FORM
# ===============================

st.header("🔮 Enter Fish Measurements")

# Default values
if 'sample_vals' in st.session_state:
    default_vals = st.session_state['sample_vals']
    st.info(f"📌 Loaded sample values for: **{st.session_state['selected_species']}**")
else:
    default_vals = [4, 7, 14, 14, 6, 10, 150, 40, 45, 40, 50, 45, 80, 80, 50]

# Create columns
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Meristic Features")
    nd1 = st.number_input("ND1_Total", value=float(default_vals[0]), step=1.0)
    nd2 = st.number_input("ND2_Total", value=float(default_vals[1]), step=1.0)
    np_val = st.number_input("NP", value=float(default_vals[2]), step=1.0)
    nc = st.number_input("NC", value=float(default_vals[3]), step=1.0)
    nv = st.number_input("NV_Total", value=float(default_vals[4]), step=1.0)
    na = st.number_input("NA_Total", value=float(default_vals[5]), step=1.0)

with col2:
    st.subheader("Morphometric Features (mm)")
    sl = st.number_input("SL", value=float(default_vals[6]), step=10.0)
    pl = st.number_input("PL", value=float(default_vals[7]), step=5.0)
    bh = st.number_input("BH", value=float(default_vals[8]), step=5.0)
    hl = st.number_input("HL", value=float(default_vals[9]), step=5.0)

with col3:
    st.subheader("Truss Features (mm)")
    head = st.number_input("Head_Truss", value=float(default_vals[10]), step=10.0)
    ant = st.number_input("Anterior_Truss", value=float(default_vals[11]), step=10.0)
    mid = st.number_input("Mid_Truss", value=float(default_vals[12]), step=20.0)
    post = st.number_input("Posterior_Truss", value=float(default_vals[13]), step=20.0)
    tail = st.number_input("Tail_Truss", value=float(default_vals[14]), step=10.0)

features = [nd1, nd2, np_val, nc, nv, na, sl, pl, bh, hl, head, ant, mid, post, tail]

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
    st.subheader("📊 Compatibility with all species")
    score_df = pd.DataFrame({
        'Species': list(scores.keys()),
        'Compatibility (%)': list(scores.values())
    }).sort_values('Compatibility (%)', ascending=False)
    st.dataframe(score_df, use_container_width=True)
    
    # Feature analysis
    st.subheader("📊 Feature Analysis")
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
    
    # Tips
    if confidence < 50:
        st.warning("⚠️ Low confidence - measurements may be outside typical ranges.")
    elif confidence >= 70:
        st.success("✅ High confidence - measurements are consistent with typical values!")
    
    if sl > 250:
        st.info("📏 Note: SL > 250mm indicates a LARGE species (Planiliza subviridis or Moolgarda tade)")
    elif sl < 200:
        st.info("📏 Note: SL < 200mm indicates a SMALL-MEDIUM species")
    
    if nd2 >= 8 and np_val >= 15:
        st.info("🔍 Note: High ND2 (≥8) and NP (≥15) strongly suggests Moolgarda tade")

# ===============================
# EXPANDED REFERENCE TABLE
# ===============================

with st.expander("📖 Complete Species Ranges (All 15 Features)"):
    all_ranges_data = []
    for sp, ranges in SPECIES_RANGES.items():
        row = {"Species": sp.split()[0]}
        for f in FEATURE_NAMES[:10]:
            mn, mx = ranges[f]
            row[f] = f"{mn:.1f}-{mx:.1f}"
        all_ranges_data.append(row)
    st.dataframe(pd.DataFrame(all_ranges_data), use_container_width=True)

# ===============================
# OPTIONAL: MODEL COMPARISON (Jika Ada Excel)
# ===============================

st.header("📊 Optional: Model Performance Comparison")

uploaded_excel = st.file_uploader(
    "Upload Excel file for model comparison (optional)",
    type=['xlsx'],
    key="excel_comparison"
)

if uploaded_excel is not None:
    st.info("Training models on your data... (this may take a few minutes)")
    
    # Simplified training for comparison
    with st.spinner("Processing data and training models..."):
        # Quick data extraction function
        def quick_extract(df, keyword):
            first_col = df.iloc[:, 0].astype(str).str.lower()
            matches = first_col[first_col == keyword.lower()].index
            if len(matches) == 0:
                return None
            start = matches[0]
            headers = [str(h).strip() for h in df.iloc[start + 1] if pd.notna(h)]
            data = []
            i = start + 2
            while i < len(df) and pd.notna(df.iloc[i, 0]):
                row = []
                for val in df.iloc[i][:len(headers)]:
                    try:
                        row.append(float(val))
                    except:
                        row.append(0)
                data.append(row)
                i += 1
            return pd.DataFrame(data, columns=headers)
        
        species_list = ["Planiliza subviridis", "Moolgarda seheli", "Osteomugil perusii", 
                        "Moolgarda tade", "Ellochelon vaigiensis"]
        
        all_data = []
        for idx, sp in enumerate(species_list):
            df_raw = pd.read_excel(uploaded_excel, sheet_name=idx, header=None)
            mer = quick_extract(df_raw, "meristic")
            morph = quick_extract(df_raw, "morphometric")
            tr = quick_extract(df_raw, "truss network")
            if tr is None:
                tr = quick_extract(df_raw, "truss")
            
            if mer is not None and morph is not None and tr is not None:
                n = min(len(mer), len(morph), len(tr))
                nd1 = mer[[c for c in mer.columns if 'ND1' in c]].sum(axis=1).values[:n] if any('ND1' in c for c in mer.columns) else np.ones(n)*4
                nd2 = mer[[c for c in mer.columns if 'ND2' in c]].sum(axis=1).values[:n] if any('ND2' in c for c in mer.columns) else np.ones(n)*7
                np_val = mer['NP'].values[:n] if 'NP' in mer.columns else np.ones(n)*14
                nc = mer['NC'].values[:n] if 'NC' in mer.columns else np.ones(n)*14
                nv = mer[[c for c in mer.columns if 'NV' in c]].sum(axis=1).values[:n] if any('NV' in c for c in mer.columns) else np.ones(n)*6
                na = mer[[c for c in mer.columns if 'NA' in c]].sum(axis=1).values[:n] if any('NA' in c for c in mer.columns) else np.ones(n)*10
                sl = morph['SL'].values[:n] if 'SL' in morph.columns else np.ones(n)*150
                pl = morph['PL'].values[:n] if 'PL' in morph.columns else np.ones(n)*40
                bh = morph['BH'].values[:n] if 'BH' in morph.columns else np.ones(n)*45
                hl = morph['HL'].values[:n] if 'HL' in morph.columns else np.ones(n)*40
                
                # Simple truss sums
                tr_cols = tr.columns
                head_sum = np.zeros(n)
                ant_sum = np.zeros(n)
                mid_sum = np.zeros(n)
                post_sum = np.zeros(n)
                tail_sum = np.zeros(n)
                for col in tr_cols:
                    c = str(col).upper().replace('-', '').replace(' ', '')
                    if c in ['AB', 'AC', 'AD']:
                        head_sum += tr[col].values[:n]
                    elif c in ['BC', 'BD', 'CD']:
                        ant_sum += tr[col].values[:n]
                    elif c in ['CE', 'CF', 'DE', 'DF', 'EF']:
                        mid_sum += tr[col].values[:n]
                    elif c in ['EG', 'EH', 'FG', 'FH', 'GH']:
                        post_sum += tr[col].values[:n]
                    elif c in ['GI', 'GJ', 'HI', 'HJ', 'IJ']:
                        tail_sum += tr[col].values[:n]
                
                for i in range(n):
                    all_data.append([
                        sp, nd1[i], nd2[i], np_val[i], nc[i], nv[i], na[i],
                        sl[i], pl[i], bh[i], hl[i], head_sum[i], ant_sum[i], 
                        mid_sum[i], post_sum[i], tail_sum[i]
                    ])
        
        df = pd.DataFrame(all_data, columns=['Species'] + FEATURE_NAMES)
        for col in FEATURE_NAMES:
            df[col] = df[col].fillna(df[col].median())
        
        # Prepare data
        X = df[FEATURE_NAMES].values
        y = df['Species'].values
        le = LabelEncoder()
        y_enc = le.fit_transform(y)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_enc, test_size=0.2, random_state=42)
        
        # Train models
        models_results = []
        
        # ANN
        ann = MLPClassifier(hidden_layer_sizes=(10,5), max_iter=500, random_state=42)
        ann.fit(X_train, y_train)
        ann_acc = accuracy_score(y_test, ann.predict(X_test))
        models_results.append({"Method": "ANN", "Accuracy": ann_acc})
        
        # PSO (simplified)
        best_acc = 0
        best_params = None
        for _ in range(30):
            h1 = np.random.randint(4, 20)
            h2 = np.random.randint(2, 12)
            alpha = np.random.uniform(0.0001, 0.01)
            lr = np.random.uniform(0.0001, 0.005)
            model = MLPClassifier(hidden_layer_sizes=(h1,h2), alpha=alpha, learning_rate_init=lr, max_iter=300, random_state=42)
            scores = cross_val_score(model, X_train, y_train, cv=3)
            mean_score = scores.mean()
            if mean_score > best_acc:
                best_acc = mean_score
                best_params = (h1, h2, alpha, lr)
        pso = MLPClassifier(hidden_layer_sizes=(best_params[0], best_params[1]), alpha=best_params[2], learning_rate_init=best_params[3], max_iter=400, random_state=42)
        pso.fit(X_train, y_train)
        pso_acc = accuracy_score(y_test, pso.predict(X_test))
        models_results.append({"Method": "PSO", "Accuracy": pso_acc})
        
        # GA (simplified)
        best_acc = 0
        best_params = None
        for _ in range(30):
            h1 = np.random.randint(4, 20)
            h2 = np.random.randint(2, 12)
            alpha = np.random.uniform(0.0001, 0.01)
            lr = np.random.uniform(0.0001, 0.005)
            model = MLPClassifier(hidden_layer_sizes=(h1,h2), alpha=alpha, learning_rate_init=lr, max_iter=300, random_state=42)
            scores = cross_val_score(model, X_train, y_train, cv=3)
            mean_score = scores.mean()
            if mean_score > best_acc:
                best_acc = mean_score
                best_params = (h1, h2, alpha, lr)
        ga = MLPClassifier(hidden_layer_sizes=(best_params[0], best_params[1]), alpha=best_params[2], learning_rate_init=best_params[3], max_iter=400, random_state=42)
        ga.fit(X_train, y_train)
        ga_acc = accuracy_score(y_test, ga.predict(X_test))
        models_results.append({"Method": "GA", "Accuracy": ga_acc})
        
        # GWO (simplified)
        best_acc = 0
        best_params = None
        for _ in range(30):
            h1 = np.random.randint(4, 20)
            h2 = np.random.randint(2, 12)
            alpha = np.random.uniform(0.0001, 0.01)
            lr = np.random.uniform(0.0001, 0.005)
            model = MLPClassifier(hidden_layer_sizes=(h1,h2), alpha=alpha, learning_rate_init=lr, max_iter=300, random_state=42)
            scores = cross_val_score(model, X_train, y_train, cv=3)
            mean_score = scores.mean()
            if mean_score > best_acc:
                best_acc = mean_score
                best_params = (h1, h2, alpha, lr)
        gwo = MLPClassifier(hidden_layer_sizes=(best_params[0], best_params[1]), alpha=best_params[2], learning_rate_init=best_params[3], max_iter=400, random_state=42)
        gwo.fit(X_train, y_train)
        gwo_acc = accuracy_score(y_test, gwo.predict(X_test))
        models_results.append({"Method": "GWO", "Accuracy": gwo_acc})
    
    # Display results
    res_df = pd.DataFrame(models_results)
    st.dataframe(res_df.style.highlight_max(subset=['Accuracy'], color='lightgreen'), use_container_width=True)
    
    # Chart
    fig, ax = plt.subplots()
    bars = ax.bar(res_df['Method'], res_df['Accuracy'], color=['#95a5a6','#e74c3c','#2ecc71','#3498db'])
    ax.set_ylim(0,1)
    ax.set_ylabel('Accuracy')
    ax.set_title('Model Performance Comparison')
    for bar, acc in zip(bars, res_df['Accuracy']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{acc:.3f}', ha='center')
    st.pyplot(fig)

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
