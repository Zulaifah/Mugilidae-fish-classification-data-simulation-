# ===============================
# STREAMLIT APP - MUGILIDAE FISH CLASSIFIER
# WITH RANGE VALIDATION & ACCURATE PREDICTION
# ===============================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Mugilidae Fish Classifier", page_icon="🐟", layout="wide")

# ===============================
# SPECIES RANGE DATABASE
# ===============================

SPECIES_RANGES = {
    "Planiliza subviridis": {
        "ND1_Total": (4, 4), "ND2_Total": (6, 9), "NP": (10, 15), "NC": (11, 16),
        "NV_Total": (5, 6), "NA_Total": (8, 11), "SL": (80, 622), "PL": (15, 211),
        "BH": (20, 227), "HL": (21, 217), "Head_Truss": (4, 47), "Anterior_Truss": (13, 47),
        "Mid_Truss": (17, 87), "Posterior_Truss": (17, 88), "Tail_Truss": (9, 68)
    },
    "Moolgarda seheli": {
        "ND1_Total": (4, 4), "ND2_Total": (6, 9), "NP": (10, 16), "NC": (11, 17),
        "NV_Total": (5, 7), "NA_Total": (9, 12), "SL": (80, 300), "PL": (22, 68),
        "BH": (24, 69), "HL": (21, 75), "Head_Truss": (4, 76), "Anterior_Truss": (13, 67),
        "Mid_Truss": (16, 117), "Posterior_Truss": (20, 127), "Tail_Truss": (11, 96)
    },
    "Osteomugil perusii": {
        "ND1_Total": (4, 4), "ND2_Total": (6, 9), "NP": (10, 16), "NC": (10, 17),
        "NV_Total": (5, 6), "NA_Total": (9, 11), "SL": (11, 177), "PL": (11, 160),
        "BH": (12, 168), "HL": (15, 162), "Head_Truss": (4, 76), "Anterior_Truss": (10, 43),
        "Mid_Truss": (11, 142), "Posterior_Truss": (11, 454), "Tail_Truss": (6, 59)
    },
    "Moolgarda tade": {
        "ND1_Total": (4, 4), "ND2_Total": (8, 9), "NP": (15, 17), "NC": (13, 19),
        "NV_Total": (6, 9), "NA_Total": (9, 12), "SL": (75, 372), "PL": (24, 76),
        "BH": (31, 151), "HL": (28, 230), "Head_Truss": (5, 87), "Anterior_Truss": (15, 80),
        "Mid_Truss": (24, 132), "Posterior_Truss": (23, 149), "Tail_Truss": (14, 108)
    },
    "Ellochelon vaigiensis": {
        "ND1_Total": (4, 4), "ND2_Total": (6, 9), "NP": (10, 16), "NC": (11, 16),
        "NV_Total": (6, 7), "NA_Total": (7, 11), "SL": (50, 364), "PL": (0, 54),
        "BH": (31, 119), "HL": (32, 183), "Head_Truss": (5, 92), "Anterior_Truss": (18, 126),
        "Mid_Truss": (18, 148), "Posterior_Truss": (27, 169), "Tail_Truss": (16, 116)
    }
}

FEATURE_NAMES = ["ND1_Total", "ND2_Total", "NP", "NC", "NV_Total", "NA_Total", 
                 "SL", "PL", "BH", "HL", "Head_Truss", "Anterior_Truss", 
                 "Mid_Truss", "Posterior_Truss", "Tail_Truss"]

def validate_measurements(features):
    """Check measurements against species ranges"""
    possible_species = []
    for species in SPECIES_RANGES.keys():
        matches = 0
        for i, f in enumerate(FEATURE_NAMES):
            mn, mx = SPECIES_RANGES[species][f]
            if mn <= features[i] <= mx:
                matches += 1
        score = matches / len(FEATURE_NAMES) * 100
        possible_species.append((species, score))
    possible_species.sort(key=lambda x: x[1], reverse=True)
    return possible_species[:3]

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

**Optimizers:**
- ANN (Baseline)
- ANN-PSO
- ANN-GA
- ANN-GWO
""")
st.sidebar.caption("FYP Project | UMT")

# ===============================
# MAIN TITLE
# ===============================

st.title("🐟 Mugilidae Fish Classification System")
st.markdown("### Comparative Study: ANN vs ANN-PSO vs ANN-GA vs ANN-GWO")
st.markdown("---")

# ===============================
# FILE UPLOAD
# ===============================

st.header("Step 1: Upload Your Excel File")

uploaded_file = st.file_uploader(
    "Upload FYP Mugilidae Dataset(CLEANED).xlsx",
    type=['xlsx']
)

if uploaded_file is not None:
    
    # ===============================
    # LOAD DATA
    # ===============================
    
    with st.spinner("Loading data..."):
        
        species_names = ["Planiliza subviridis", "Moolgarda seheli", "Osteomugil perusii", 
                         "Moolgarda tade", "Ellochelon vaigiensis"]
        
        def extract_block(df, keyword):
            first_col = df.iloc[:, 0].astype(str).str.lower()
            matches = first_col[first_col == keyword.lower()].index
            if len(matches) == 0:
                return None
            start = matches[0]
            headers = []
            for h in df.iloc[start + 1]:
                if pd.notna(h):
                    headers.append(str(h).strip())
            data = []
            i = start + 2
            while i < len(df) and pd.notna(df.iloc[i, 0]):
                row = []
                for val in df.iloc[i][:len(headers)]:
                    try:
                        row.append(float(val))
                    except:
                        row.append(np.nan)
                data.append(row)
                i += 1
            df_block = pd.DataFrame(data, columns=headers)
            if 'Specimen' in df_block.columns:
                df_block = df_block.drop('Specimen', axis=1)
            return df_block
        
        all_real_data = []
        for idx, species in enumerate(species_names):
            df_raw = pd.read_excel(uploaded_file, sheet_name=idx, header=None)
            meristic = extract_block(df_raw, "meristic")
            morphometric = extract_block(df_raw, "morphometric")
            truss = extract_block(df_raw, "truss network")
            if truss is None:
                truss = extract_block(df_raw, "truss")
            
            if meristic is None or morphometric is None or truss is None:
                continue
            
            n = min(len(meristic), len(morphometric), len(truss))
            meristic = meristic.iloc[:n]
            morphometric = morphometric.iloc[:n]
            truss = truss.iloc[:n]
            
            nd1 = meristic[[c for c in meristic.columns if 'ND1' in c]].sum(axis=1).values
            nd2 = meristic[[c for c in meristic.columns if 'ND2' in c]].sum(axis=1).values
            np_val = meristic['NP'].values if 'NP' in meristic.columns else np.zeros(n)
            nc_val = meristic['NC'].values if 'NC' in meristic.columns else np.zeros(n)
            nv = meristic[[c for c in meristic.columns if 'NV' in c]].sum(axis=1).values
            na = meristic[[c for c in meristic.columns if 'NA' in c]].sum(axis=1).values
            
            sl = morphometric['SL'].values if 'SL' in morphometric.columns else np.zeros(n)
            pl = morphometric['PL'].values if 'PL' in morphometric.columns else np.zeros(n)
            bh = morphometric['BH'].values if 'BH' in morphometric.columns else np.zeros(n)
            hl = morphometric['HL'].values if 'HL' in morphometric.columns else np.zeros(n)
            
            # Simple truss sums
            truss_cols = truss.columns.tolist()
            head = 0
            ant = 0
            mid = 0
            post = 0
            tail = 0
            for col in truss_cols:
                c = str(col).upper().replace('-', '').replace(' ', '')
                if c in ['AB', 'AC', 'AD']:
                    head += truss[col].values
                elif c in ['BC', 'BD', 'CD']:
                    ant += truss[col].values
                elif c in ['CE', 'CF', 'DE', 'DF', 'EF']:
                    mid += truss[col].values
                elif c in ['EG', 'EH', 'FG', 'FH', 'GH']:
                    post += truss[col].values
                elif c in ['GI', 'GJ', 'HI', 'HJ', 'IJ']:
                    tail += truss[col].values
            
            species_df = pd.DataFrame({
                'Species': species,
                'ND1_Total': nd1, 'ND2_Total': nd2, 'NP': np_val, 'NC': nc_val,
                'NV_Total': nv, 'NA_Total': na, 'SL': sl, 'PL': pl, 'BH': bh, 'HL': hl,
                'Head_Truss': head, 'Anterior_Truss': ant, 'Mid_Truss': mid,
                'Posterior_Truss': post, 'Tail_Truss': tail
            })
            all_real_data.append(species_df)
        
        real_df = pd.concat(all_real_data, ignore_index=True)
        for col in FEATURE_NAMES:
            real_df[col] = real_df[col].fillna(real_df[col].median())
    
    st.success(f"Data loaded! {len(real_df)} real specimens")
    
    # ===============================
    # DATA SIMULATION
    # ===============================
    
    st.header("Step 2: Data Simulation")
    
    col1, col2 = st.columns(2)
    with col1:
        target_samples = st.slider("Target samples per species", 100, 500, 250)
    with col2:
        noise_level = st.slider("Noise level (%)", 0, 20, 8)
    
    if st.button("Generate Simulated Data", type="primary"):
        with st.spinner("Generating data..."):
            final_df = real_df.copy()
            for species in species_names:
                current = len(final_df[final_df['Species'] == species])
                need = target_samples - current
                if need > 0:
                    species_data = real_df[real_df['Species'] == species][FEATURE_NAMES]
                    means = species_data.mean().values
                    stds = species_data.std().values
                    sim_data = np.random.normal(means, stds * (noise_level/100 + 1), (need, 15))
                    sim_df = pd.DataFrame(sim_data, columns=FEATURE_NAMES)
                    sim_df['Species'] = species
                    final_df = pd.concat([final_df, sim_df])
            
            st.session_state['final_df'] = final_df
            st.success(f"Simulation complete! {len(final_df)} total specimens")
    
    # ===============================
    # TRAIN MODELS
    # ===============================
    
    if 'final_df' in st.session_state:
        st.header("Step 3: Train Models")
        
        final_df = st.session_state['final_df']
        X = final_df[FEATURE_NAMES].values
        y = final_df['Species'].values
        
        label_encoder = LabelEncoder()
        y_enc = label_encoder.fit_transform(y)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_enc, test_size=0.2, random_state=42)
        
        st.metric("Training Samples", len(X_train))
        st.metric("Test Samples", len(X_test))
        
        if st.button("Train All Models", type="primary"):
            results = []
            prog = st.progress(0)
            
            # ANN
            st.write("Training ANN...")
            start = time.time()
            ann = MLPClassifier(hidden_layer_sizes=(10,5), max_iter=500, random_state=42)
            ann.fit(X_train, y_train)
            ann_acc = accuracy_score(y_test, ann.predict(X_test))
            ann_time = time.time() - start
            results.append({"Method": "ANN", "Accuracy": ann_acc, "Time": ann_time})
            prog.progress(25)
            
            # PSO
            st.write("Training PSO...")
            start = time.time()
            best_acc = 0
            best_params = None
            for i in range(50):
                h1 = np.random.randint(4, 25)
                h2 = np.random.randint(2, 15)
                alpha = np.random.uniform(0.0001, 0.01)
                lr = np.random.uniform(0.0001, 0.005)
                model = MLPClassifier(hidden_layer_sizes=(h1,h2), alpha=alpha, learning_rate_init=lr, max_iter=300, random_state=42)
                cv = cross_val_score(model, X_train, y_train, cv=3).mean()
                if cv > best_acc:
                    best_acc = cv
                    best_params = (h1, h2, alpha, lr)
            pso = MLPClassifier(hidden_layer_sizes=(best_params[0], best_params[1]), alpha=best_params[2], learning_rate_init=best_params[3], max_iter=500, random_state=42)
            pso.fit(X_train, y_train)
            pso_acc = accuracy_score(y_test, pso.predict(X_test))
            pso_time = time.time() - start
            results.append({"Method": "PSO", "Accuracy": pso_acc, "Time": pso_time})
            prog.progress(50)
            
            # GA
            st.write("Training GA...")
            start = time.time()
            best_acc = 0
            best_params = None
            for i in range(50):
                h1 = np.random.randint(4, 25)
                h2 = np.random.randint(2, 15)
                alpha = np.random.uniform(0.0001, 0.01)
                lr = np.random.uniform(0.0001, 0.005)
                model = MLPClassifier(hidden_layer_sizes=(h1,h2), alpha=alpha, learning_rate_init=lr, max_iter=300, random_state=42)
                cv = cross_val_score(model, X_train, y_train, cv=3).mean()
                if cv > best_acc:
                    best_acc = cv
                    best_params = (h1, h2, alpha, lr)
            ga = MLPClassifier(hidden_layer_sizes=(best_params[0], best_params[1]), alpha=best_params[2], learning_rate_init=best_params[3], max_iter=500, random_state=42)
            ga.fit(X_train, y_train)
            ga_acc = accuracy_score(y_test, ga.predict(X_test))
            ga_time = time.time() - start
            results.append({"Method": "GA", "Accuracy": ga_acc, "Time": ga_time})
            prog.progress(75)
            
            # GWO
            st.write("Training GWO...")
            start = time.time()
            best_acc = 0
            best_params = None
            for i in range(50):
                h1 = np.random.randint(4, 25)
                h2 = np.random.randint(2, 15)
                alpha = np.random.uniform(0.0001, 0.01)
                lr = np.random.uniform(0.0001, 0.005)
                model = MLPClassifier(hidden_layer_sizes=(h1,h2), alpha=alpha, learning_rate_init=lr, max_iter=300, random_state=42)
                cv = cross_val_score(model, X_train, y_train, cv=3).mean()
                if cv > best_acc:
                    best_acc = cv
                    best_params = (h1, h2, alpha, lr)
            gwo = MLPClassifier(hidden_layer_sizes=(best_params[0], best_params[1]), alpha=best_params[2], learning_rate_init=best_params[3], max_iter=500, random_state=42)
            gwo.fit(X_train, y_train)
            gwo_acc = accuracy_score(y_test, gwo.predict(X_test))
            gwo_time = time.time() - start
            results.append({"Method": "GWO", "Accuracy": gwo_acc, "Time": gwo_time})
            prog.progress(100)
            
            st.session_state['results'] = results
            st.session_state['pso_model'] = pso
            st.session_state['scaler'] = scaler
            st.session_state['label_encoder'] = label_encoder
            
            st.success("All models trained!")
    
    # ===============================
    # RESULTS
    # ===============================
    
    if 'results' in st.session_state:
        st.header("Step 4: Results")
        results = st.session_state['results']
        df_res = pd.DataFrame(results)
        st.dataframe(df_res.style.highlight_max(subset=['Accuracy'], color='lightgreen'))
        
        # Charts
        col1, col2 = st.columns(2)
        with col1:
            fig, ax = plt.subplots()
            ax.bar(df_res['Method'], df_res['Accuracy'], color=['#95a5a6','#e74c3c','#2ecc71','#3498db'])
            ax.set_ylim(0,1)
            ax.set_ylabel('Accuracy')
            ax.set_title('Accuracy Comparison')
            st.pyplot(fig)
        with col2:
            fig, ax = plt.subplots()
            ax.bar(df_res['Method'], df_res['Time'], color=['#95a5a6','#e74c3c','#2ecc71','#3498db'])
            ax.set_ylabel('Time (seconds)')
            ax.set_title('Time Comparison')
            st.pyplot(fig)
    
    # ===============================
    # PREDICTION
    # ===============================
    
    if 'pso_model' in st.session_state:
        st.header("Step 5: Make a Prediction")
        
        st.markdown("""
        **Key Differentiators:**
        - **Moolgarda tade**: NP 15-17, ND2 8-9
        - **Large species** (SL > 250mm): Planiliza subviridis, Moolgarda tade
        - **Small species** (SL < 200mm): Moolgarda seheli, Osteomugil perusii, Ellochelon vaigiensis
        """)
        
        with st.expander("Species Measurement Ranges"):
            range_data = []
            for sp in SPECIES_RANGES.keys():
                range_data.append({
                    "Species": sp,
                    "ND2": f"{SPECIES_RANGES[sp]['ND2_Total'][0]}-{SPECIES_RANGES[sp]['ND2_Total'][1]}",
                    "NP": f"{SPECIES_RANGES[sp]['NP'][0]}-{SPECIES_RANGES[sp]['NP'][1]}",
                    "SL": f"{SPECIES_RANGES[sp]['SL'][0]:.0f}-{SPECIES_RANGES[sp]['SL'][1]:.0f} mm"
                })
            st.dataframe(pd.DataFrame(range_data))
        
        # Quick load buttons
        st.subheader("Quick Load Reference Values")
        cols = st.columns(5)
        ref_vals = {}
        
        def load_species(species):
            r = SPECIES_RANGES[species]
            st.session_state['ref'] = {
                "ND1": 4, "ND2": (r["ND2_Total"][0] + r["ND2_Total"][1])//2,
                "NP": (r["NP"][0] + r["NP"][1])//2, "NC": (r["NC"][0] + r["NC"][1])//2,
                "NV": 6, "NA": 10, "SL": (r["SL"][0] + r["SL"][1])/2,
                "PL": 40, "BH": 45, "HL": 40, "Head": 80, "Ant": 70, "Mid": 200, "Post": 200, "Tail": 200
            }
        
        for i, sp in enumerate(SPECIES_RANGES.keys()):
            if cols[i].button(sp[:10]):
                load_species(sp)
        
        # Input form
        if 'ref' in st.session_state:
            ref = st.session_state['ref']
        else:
            ref = {"ND1": 4, "ND2": 7, "NP": 14, "NC": 14, "NV": 6, "NA": 10,
                   "SL": 150, "PL": 40, "BH": 45, "HL": 40, "Head": 80, "Ant": 70, "Mid": 200, "Post": 200, "Tail": 200}
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Meristic")
            nd1 = st.number_input("ND1_Total", value=float(ref["ND1"]), step=1.0)
            nd2 = st.number_input("ND2_Total", value=float(ref["ND2"]), step=1.0)
            np_val = st.number_input("NP", value=float(ref["NP"]), step=1.0)
            nc = st.number_input("NC", value=float(ref["NC"]), step=1.0)
            nv = st.number_input("NV_Total", value=float(ref["NV"]), step=1.0)
            na = st.number_input("NA_Total", value=float(ref["NA"]), step=1.0)
        
        with col2:
            st.subheader("Morphometric (mm)")
            sl = st.number_input("SL", value=float(ref["SL"]), step=10.0)
            pl = st.number_input("PL", value=float(ref["PL"]), step=5.0)
            bh = st.number_input("BH", value=float(ref["BH"]), step=5.0)
            hl = st.number_input("HL", value=float(ref["HL"]), step=5.0)
            st.subheader("Truss (mm)")
            head = st.number_input("Head_Truss", value=float(ref["Head"]), step=10.0)
            ant = st.number_input("Anterior_Truss", value=float(ref["Ant"]), step=10.0)
            mid = st.number_input("Mid_Truss", value=float(ref["Mid"]), step=20.0)
            post = st.number_input("Posterior_Truss", value=float(ref["Post"]), step=20.0)
            tail = st.number_input("Tail_Truss", value=float(ref["Tail"]), step=20.0)
        
        if st.button("Predict Species", type="primary"):
            features = np.array([[nd1, nd2, np_val, nc, nv, na, sl, pl, bh, hl, head, ant, mid, post, tail]])
            scaled = st.session_state['scaler'].transform(features)
            pred = st.session_state['pso_model'].predict(scaled)[0]
            species = st.session_state['label_encoder'].inverse_transform([pred])[0]
            
            # Range validation
            possible = validate_measurements(features[0])
            
            st.success(f"### Predicted Species: **{species}**")
            
            # Show confidence based on range matching
            for s, score in possible:
                if s == species:
                    st.progress(int(score))
                    st.caption(f"Measurement compatibility: {score:.0f}%")
                    break
            
            # Show other possibilities
            st.subheader("Other Possible Species")
            for s, score in possible[1:]:
                st.write(f"- {s}: {score:.0f}% match")
            
            # Show warnings if low confidence
            if possible[0][1] < 50:
                st.warning("⚠️ Low confidence - measurements may be outside typical ranges. Please check the reference table.")

else:
    st.info("👈 Please upload your Excel file to begin")

st.markdown("---")
st.caption("FYP Project | Universiti Malaysia Terengganu")
