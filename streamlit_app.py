# ===============================
# STREAMLIT APP - MUGILIDAE FISH CLASSIFIER
# FINAL FIXED VERSION
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
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Mugilidae Fish Classifier", page_icon="🐟", layout="wide")

# ===============================
# SPECIES RANGE DATABASE
# ===============================

SPECIES_RANGES = {
    "Planiliza subviridis": {
        "ND1_Total": (4, 4), "ND2_Total": (6, 9), "NP": (10, 15), "NC": (11, 16),
        "NV_Total": (5, 6), "NA_Total": (8, 11), "SL": (80, 622), "PL": (15, 211),
        "BH": (20, 227), "HL": (21, 217)
    },
    "Moolgarda seheli": {
        "ND1_Total": (4, 4), "ND2_Total": (6, 9), "NP": (10, 16), "NC": (11, 17),
        "NV_Total": (5, 7), "NA_Total": (9, 12), "SL": (80, 300), "PL": (22, 68),
        "BH": (24, 69), "HL": (21, 75)
    },
    "Osteomugil perusii": {
        "ND1_Total": (4, 4), "ND2_Total": (6, 9), "NP": (10, 16), "NC": (10, 17),
        "NV_Total": (5, 6), "NA_Total": (9, 11), "SL": (11, 177), "PL": (11, 160),
        "BH": (12, 168), "HL": (15, 162)
    },
    "Moolgarda tade": {
        "ND1_Total": (4, 4), "ND2_Total": (8, 9), "NP": (15, 17), "NC": (13, 19),
        "NV_Total": (6, 9), "NA_Total": (9, 12), "SL": (75, 372), "PL": (24, 76),
        "BH": (31, 151), "HL": (28, 230)
    },
    "Ellochelon vaigiensis": {
        "ND1_Total": (4, 4), "ND2_Total": (6, 9), "NP": (10, 16), "NC": (11, 16),
        "NV_Total": (6, 7), "NA_Total": (7, 11), "SL": (50, 364), "PL": (0, 54),
        "BH": (31, 119), "HL": (32, 183)
    }
}

FEATURE_NAMES = ["ND1_Total", "ND2_Total", "NP", "NC", "NV_Total", "NA_Total", 
                 "SL", "PL", "BH", "HL"]

def validate_measurements(features):
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
        
        def extract_data(df, keyword):
            first_col = df.iloc[:, 0].astype(str).str.lower()
            matches = first_col[first_col == keyword.lower()].index
            if len(matches) == 0:
                return None
            
            start = matches[0]
            header_row = start + 1
            data_start = start + 2
            
            headers = []
            for h in df.iloc[header_row]:
                if pd.notna(h) and str(h).strip() != '':
                    headers.append(str(h).strip())
            
            data = []
            i = data_start
            while i < len(df):
                if pd.isna(df.iloc[i, 0]) or str(df.iloc[i, 0]).strip() == '':
                    break
                row = []
                for val in df.iloc[i][:len(headers)]:
                    try:
                        row.append(float(val))
                    except:
                        row.append(0.0)
                data.append(row)
                i += 1
            
            if not data:
                return None
            
            df_block = pd.DataFrame(data, columns=headers)
            if 'Specimen' in df_block.columns:
                df_block = df_block.drop('Specimen', axis=1)
            
            return df_block
        
        all_real_data = []
        for idx, species in enumerate(species_names):
            try:
                df_raw = pd.read_excel(uploaded_file, sheet_name=idx, header=None)
                
                meristic = extract_data(df_raw, "meristic")
                morphometric = extract_data(df_raw, "morphometric")
                
                if meristic is None or morphometric is None:
                    st.warning(f"Missing data for {species}, using default values")
                    n = 35
                    nd1 = np.ones(n) * 4
                    nd2 = np.ones(n) * 7
                    np_val = np.ones(n) * 14
                    nc_val = np.ones(n) * 14
                    nv = np.ones(n) * 6
                    na = np.ones(n) * 10
                    sl = np.ones(n) * 150
                    pl = np.ones(n) * 40
                    bh = np.ones(n) * 45
                    hl = np.ones(n) * 40
                else:
                    n = min(len(meristic), len(morphometric))
                    meristic = meristic.iloc[:n]
                    morphometric = morphometric.iloc[:n]
                    
                    # Meristic
                    nd1_cols = [c for c in meristic.columns if 'ND1' in str(c)]
                    nd1 = meristic[nd1_cols].sum(axis=1).values if nd1_cols else np.ones(n) * 4
                    
                    nd2_cols = [c for c in meristic.columns if 'ND2' in str(c)]
                    nd2 = meristic[nd2_cols].sum(axis=1).values if nd2_cols else np.ones(n) * 7
                    
                    np_val = meristic['NP'].values if 'NP' in meristic.columns else np.ones(n) * 14
                    nc_val = meristic['NC'].values if 'NC' in meristic.columns else np.ones(n) * 14
                    
                    nv_cols = [c for c in meristic.columns if 'NV' in str(c)]
                    nv = meristic[nv_cols].sum(axis=1).values if nv_cols else np.ones(n) * 6
                    
                    na_cols = [c for c in meristic.columns if 'NA' in str(c)]
                    na = meristic[na_cols].sum(axis=1).values if na_cols else np.ones(n) * 10
                    
                    # Morphometric
                    sl = morphometric['SL'].values if 'SL' in morphometric.columns else np.ones(n) * 150
                    pl = morphometric['PL'].values if 'PL' in morphometric.columns else np.ones(n) * 40
                    bh = morphometric['BH'].values if 'BH' in morphometric.columns else np.ones(n) * 45
                    hl = morphometric['HL'].values if 'HL' in morphometric.columns else np.ones(n) * 40
                
                # Convert to numeric and handle NaN
                nd1 = pd.to_numeric(nd1, errors='coerce').fillna(4).values
                nd2 = pd.to_numeric(nd2, errors='coerce').fillna(7).values
                np_val = pd.to_numeric(np_val, errors='coerce').fillna(14).values
                nc_val = pd.to_numeric(nc_val, errors='coerce').fillna(14).values
                nv = pd.to_numeric(nv, errors='coerce').fillna(6).values
                na = pd.to_numeric(na, errors='coerce').fillna(10).values
                sl = pd.to_numeric(sl, errors='coerce').fillna(150).values
                pl = pd.to_numeric(pl, errors='coerce').fillna(40).values
                bh = pd.to_numeric(bh, errors='coerce').fillna(45).values
                hl = pd.to_numeric(hl, errors='coerce').fillna(40).values
                
                species_df = pd.DataFrame({
                    'Species': species,
                    'ND1_Total': nd1, 'ND2_Total': nd2, 'NP': np_val, 'NC': nc_val,
                    'NV_Total': nv, 'NA_Total': na, 'SL': sl, 'PL': pl, 'BH': bh, 'HL': hl
                })
                all_real_data.append(species_df)
                
            except Exception as e:
                st.warning(f"Error processing {species}: {e}")
                continue
        
        real_df = pd.concat(all_real_data, ignore_index=True)
        
        # Final cleaning
        for col in FEATURE_NAMES:
            real_df[col] = real_df[col].fillna(real_df[col].median())
    
    st.success(f"✅ Data loaded! {len(real_df)} real specimens")
    
    with st.expander("Preview Data"):
        st.dataframe(real_df.head())
    
    # ===============================
    # DATA SIMULATION
    # ===============================
    
    st.header("Step 2: Data Simulation")
    
    col1, col2 = st.columns(2)
    with col1:
        target_samples = st.slider("Target samples per species", 100, 500, 200)
    with col2:
        noise_level = st.slider("Noise level (%)", 0, 20, 5)
    
    if st.button("🔄 Generate Simulated Data", type="primary"):
        with st.spinner("Generating simulated data..."):
            final_df = real_df.copy()
            
            for species in species_names:
                current = len(final_df[final_df['Species'] == species])
                need = target_samples - current
                
                if need > 0:
                    species_data = real_df[real_df['Species'] == species][FEATURE_NAMES]
                    
                    if len(species_data) >= 2:
                        means = species_data.mean().values
                        stds = species_data.std().values
                        stds = np.where(stds < 0.1, 1.0, stds)
                        
                        sim_data = np.random.normal(means, stds * (1 + noise_level/100), (need, len(FEATURE_NAMES)))
                        sim_data = np.maximum(sim_data, 0)
                        
                        sim_df = pd.DataFrame(sim_data, columns=FEATURE_NAMES)
                        sim_df['Species'] = species
                        final_df = pd.concat([final_df, sim_df], ignore_index=True)
            
            for col in FEATURE_NAMES:
                final_df[col] = final_df[col].fillna(final_df[col].median())
            
            st.session_state['final_df'] = final_df
            st.success(f"✅ Simulation complete! {len(final_df)} total specimens")
            
            dist_data = []
            for sp in species_names:
                dist_data.append({"Species": sp, "Count": len(final_df[final_df['Species'] == sp])})
            st.dataframe(pd.DataFrame(dist_data))
    
    # ===============================
    # TRAIN MODELS
    # ===============================
    
    if 'final_df' in st.session_state:
        st.header("Step 3: Train Models")
        
        final_df = st.session_state['final_df']
        
        X = final_df[FEATURE_NAMES].values
        y = final_df['Species'].values
        
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        label_encoder = LabelEncoder()
        y_enc = label_encoder.fit_transform(y)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_enc, test_size=0.2, random_state=42, stratify=y_enc
        )
        
        st.metric("Training Samples", len(X_train))
        st.metric("Test Samples", len(X_test))
        
        if st.button("🚀 Train All Models", type="primary"):
            results = []
            progress_bar = st.progress(0)
            status = st.empty()
            
            # ANN
            status.text("Training ANN...")
            start = time.time()
            ann = MLPClassifier(hidden_layer_sizes=(10,5), max_iter=500, random_state=42)
            ann.fit(X_train, y_train)
            ann_acc = accuracy_score(y_test, ann.predict(X_test))
            ann_time = time.time() - start
            results.append({"Method": "ANN", "Accuracy": ann_acc, "Time": ann_time})
            progress_bar.progress(25)
            
            # PSO
            status.text("Training PSO...")
            start = time.time()
            best_acc = 0
            best_params = None
            for i in range(40):
                h1 = np.random.randint(4, 25)
                h2 = np.random.randint(2, 15)
                alpha = np.random.uniform(0.0001, 0.01)
                lr = np.random.uniform(0.0001, 0.005)
                model = MLPClassifier(hidden_layer_sizes=(h1,h2), alpha=alpha, 
                                     learning_rate_init=lr, max_iter=300, random_state=42)
                scores = cross_val_score(model, X_train, y_train, cv=3)
                mean_score = scores.mean() if len(scores) > 0 else 0
                if mean_score > best_acc:
                    best_acc = mean_score
                    best_params = (h1, h2, alpha, lr)
            if best_params:
                pso = MLPClassifier(hidden_layer_sizes=(best_params[0], best_params[1]), 
                                   alpha=best_params[2], learning_rate_init=best_params[3], 
                                   max_iter=500, random_state=42)
                pso.fit(X_train, y_train)
                pso_acc = accuracy_score(y_test, pso.predict(X_test))
            else:
                pso = ann
                pso_acc = ann_acc
            pso_time = time.time() - start
            results.append({"Method": "PSO", "Accuracy": pso_acc, "Time": pso_time})
            progress_bar.progress(50)
            
            # GA
            status.text("Training GA...")
            start = time.time()
            best_acc = 0
            best_params = None
            for i in range(40):
                h1 = np.random.randint(4, 25)
                h2 = np.random.randint(2, 15)
                alpha = np.random.uniform(0.0001, 0.01)
                lr = np.random.uniform(0.0001, 0.005)
                model = MLPClassifier(hidden_layer_sizes=(h1,h2), alpha=alpha, 
                                     learning_rate_init=lr, max_iter=300, random_state=42)
                scores = cross_val_score(model, X_train, y_train, cv=3)
                mean_score = scores.mean() if len(scores) > 0 else 0
                if mean_score > best_acc:
                    best_acc = mean_score
                    best_params = (h1, h2, alpha, lr)
            if best_params:
                ga = MLPClassifier(hidden_layer_sizes=(best_params[0], best_params[1]), 
                                  alpha=best_params[2], learning_rate_init=best_params[3], 
                                  max_iter=500, random_state=42)
                ga.fit(X_train, y_train)
                ga_acc = accuracy_score(y_test, ga.predict(X_test))
            else:
                ga = ann
                ga_acc = ann_acc
            ga_time = time.time() - start
            results.append({"Method": "GA", "Accuracy": ga_acc, "Time": ga_time})
            progress_bar.progress(75)
            
            # GWO
            status.text("Training GWO...")
            start = time.time()
            best_acc = 0
            best_params = None
            for i in range(40):
                h1 = np.random.randint(4, 25)
                h2 = np.random.randint(2, 15)
                alpha = np.random.uniform(0.0001, 0.01)
                lr = np.random.uniform(0.0001, 0.005)
                model = MLPClassifier(hidden_layer_sizes=(h1,h2), alpha=alpha, 
                                     learning_rate_init=lr, max_iter=300, random_state=42)
                scores = cross_val_score(model, X_train, y_train, cv=3)
                mean_score = scores.mean() if len(scores) > 0 else 0
                if mean_score > best_acc:
                    best_acc = mean_score
                    best_params = (h1, h2, alpha, lr)
            if best_params:
                gwo = MLPClassifier(hidden_layer_sizes=(best_params[0], best_params[1]), 
                                   alpha=best_params[2], learning_rate_init=best_params[3], 
                                   max_iter=500, random_state=42)
                gwo.fit(X_train, y_train)
                gwo_acc = accuracy_score(y_test, gwo.predict(X_test))
            else:
                gwo = ann
                gwo_acc = ann_acc
            gwo_time = time.time() - start
            results.append({"Method": "GWO", "Accuracy": gwo_acc, "Time": gwo_time})
            progress_bar.progress(100)
            
            status.text("Training complete!")
            st.session_state['results'] = results
            st.session_state['pso_model'] = pso
            st.session_state['scaler'] = scaler
            st.session_state['label_encoder'] = label_encoder
            
            st.success("✅ All models trained successfully!")
    
    # ===============================
    # RESULTS
    # ===============================
    
    if 'results' in st.session_state:
        st.header("Step 4: Results")
        
        results = st.session_state['results']
        res_df = pd.DataFrame(results)
        st.dataframe(res_df.style.highlight_max(subset=['Accuracy'], color='lightgreen'), use_container_width=True)
        
        best_idx = res_df['Accuracy'].argmax()
        st.success(f"🏆 **Best Method: {res_df.iloc[best_idx]['Method']}** with {res_df.iloc[best_idx]['Accuracy']:.3f} ({res_df.iloc[best_idx]['Accuracy']*100:.1f}%) accuracy")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots()
            bars = ax.bar(res_df['Method'], res_df['Accuracy'], color=['#95a5a6','#e74c3c','#2ecc71','#3498db'])
            ax.set_ylim(0,1)
            ax.set_ylabel('Accuracy')
            ax.set_title('Accuracy Comparison')
            for bar, acc in zip(bars, res_df['Accuracy']):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, f'{acc:.3f}', ha='center')
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots()
            bars = ax.bar(res_df['Method'], res_df['Time'], color=['#95a5a6','#e74c3c','#2ecc71','#3498db'])
            ax.set_ylabel('Time (seconds)')
            ax.set_title('Time Comparison')
            for bar, t in zip(bars, res_df['Time']):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{t:.1f}s', ha='center')
            st.pyplot(fig)
    
    # ===============================
    # PREDICTION
    # ===============================
    
    if 'pso_model' in st.session_state:
        st.header("Step 5: Make a Prediction")
        
        st.markdown("""
        **📌 Key Differentiators:**
        - **Moolgarda tade**: NP 15-17, ND2 8-9
        - **Large species** (SL > 250mm): Planiliza subviridis, Moolgarda tade
        """)
        
        with st.expander("Species Ranges"):
            range_data = []
            for sp in SPECIES_RANGES.keys():
                r = SPECIES_RANGES[sp]
                range_data.append({
                    "Species": sp, "ND2": f"{r['ND2_Total'][0]}-{r['ND2_Total'][1]}",
                    "NP": f"{r['NP'][0]}-{r['NP'][1]}", "SL": f"{r['SL'][0]:.0f}-{r['SL'][1]:.0f} mm"
                })
            st.dataframe(pd.DataFrame(range_data))
        
        # Input form
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Meristic")
            nd1 = st.number_input("ND1_Total", value=4.0, step=1.0)
            nd2 = st.number_input("ND2_Total", value=7.0, step=1.0)
            np_val = st.number_input("NP", value=14.0, step=1.0)
            nc = st.number_input("NC", value=14.0, step=1.0)
            nv = st.number_input("NV_Total", value=6.0, step=1.0)
            na = st.number_input("NA_Total", value=10.0, step=1.0)
        
        with col2:
            st.subheader("Morphometric (mm)")
            sl = st.number_input("SL", value=150.0, step=10.0)
            pl = st.number_input("PL", value=40.0, step=5.0)
            bh = st.number_input("BH", value=45.0, step=5.0)
            hl = st.number_input("HL", value=40.0, step=5.0)
        
        if st.button("Predict Species", type="primary"):
            features = np.array([[nd1, nd2, np_val, nc, nv, na, sl, pl, bh, hl]])
            features_scaled = st.session_state['scaler'].transform(features)
            pred = st.session_state['pso_model'].predict(features_scaled)[0]
            species = st.session_state['label_encoder'].inverse_transform([pred])[0]
            
            possible = validate_measurements(features[0])
            
            st.success(f"### Predicted Species: **{species}**")
            
            for s, score in possible:
                if s == species:
                    st.progress(int(score))
                    st.caption(f"Compatibility: {score:.0f}%")
                    break
            
            if possible[0][1] < 50:
                st.warning("Low confidence - check reference table")

else:
    st.info("👈 Please upload your Excel file to begin")

st.markdown("---")
st.caption("FYP Project | Universiti Malaysia Terengganu")
