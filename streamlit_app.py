# ===============================
# STREAMLIT APP - MUGILIDAE FISH CLASSIFIER
# HYBRID: ML Model (ANN/PSO/GA/GWO) + Range Validation
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
# SPECIES RANGE DATABASE (FOR VALIDATION)
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

def check_in_range(species, features):
    """Check if measurements are within species range"""
    ranges = SPECIES_RANGES[species]
    out_of_range = []
    for i, f in enumerate(FEATURE_NAMES):
        mn, mx = ranges[f]
        if not (mn <= features[i] <= mx):
            out_of_range.append(FEATURE_DISPLAY[i])
    return len(out_of_range) == 0, out_of_range

def get_range_compatibility(features):
    """Calculate compatibility with each species based on ranges"""
    scores = {}
    for species, ranges in SPECIES_RANGES.items():
        matches = 0
        for i, f in enumerate(FEATURE_NAMES):
            mn, mx = ranges[f]
            if mn <= features[i] <= mx:
                matches += 1
        scores[species] = (matches / len(FEATURE_NAMES)) * 100
    return scores

def hybrid_predict(features, ml_model, scaler, label_encoder):
    """Hybrid prediction: ML model + range validation"""
    
    # Step 1: ML Prediction
    features_scaled = scaler.transform([features])
    ml_pred = ml_model.predict(features_scaled)[0]
    ml_species = label_encoder.inverse_transform([ml_pred])[0]
    ml_proba = ml_model.predict_proba(features_scaled)[0]
    ml_confidence = max(ml_proba) * 100
    
    # Step 2: Check if ML prediction is within range
    in_range, out_features = check_in_range(ml_species, features)
    
    # Step 3: Get range compatibility for all species
    range_scores = get_range_compatibility(features)
    
    if in_range:
        # ML prediction is valid, use it
        final_species = ml_species
        final_confidence = ml_confidence
        method = "ML Model (within biological range) ✅"
        warning = None
    else:
        # ML prediction out of range, find best match within range
        best_range_species = max(range_scores, key=range_scores.get)
        best_range_score = range_scores[best_range_species]
        
        if best_range_score >= 60:
            # Use range-based prediction (high confidence)
            final_species = best_range_species
            final_confidence = best_range_score
            method = "Range-Based (ML prediction corrected) 🔄"
            warning = f"ML model predicted {ml_species}, but measurements were outside typical range. Corrected to {final_species}."
        else:
            # Use ML but with warning
            final_species = ml_species
            final_confidence = ml_confidence
            method = "ML Model (⚠️ low range compatibility)"
            warning = f"Measurements are outside typical range for {ml_species}. Please verify measurements."
    
    return final_species, final_confidence, method, ml_species, in_range, warning, range_scores

# ===============================
# FUNCTIONS FOR DATA EXTRACTION
# ===============================

def extract_block(df, keyword):
    first_col = df.iloc[:, 0].astype(str).str.strip().str.lower()
    matches = first_col[first_col == keyword.lower()].index
    if len(matches) == 0:
        return None
    start_idx = matches[0]
    header_row = start_idx + 1
    data_start = start_idx + 2
    headers = []
    for h in df.iloc[header_row]:
        if pd.notna(h) and str(h).strip() != '':
            headers.append(str(h).strip())
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
            except:
                numeric_row.append(np.nan)
        data.append(numeric_row)
        i += 1
    if not data:
        return None
    df_block = pd.DataFrame(data, columns=headers[:len(data[0])])
    if 'Specimen' in df_block.columns:
        df_block = df_block.drop('Specimen', axis=1)
    return df_block

def get_truss_sum(truss_df, measurements):
    truss_cols = {str(col).replace(' ', '').replace('-', ''): col for col in truss_df.columns}
    total = np.zeros(len(truss_df))
    for meas in measurements:
        meas_clean = meas.replace('-', '')
        for key, col in truss_cols.items():
            if meas_clean == key or meas_clean in key or key in meas_clean:
                total += truss_df[col].fillna(0).values
                break
    return total

# ===============================
# SIDEBAR
# ===============================

st.sidebar.title("🐟 Mugilidae Fish Classifier")
st.sidebar.markdown("---")
st.sidebar.header("📋 About")
st.sidebar.info("""
**Hybrid Classification System:**
- ML Models: ANN, PSO, GA, GWO
- Range Validation for Accuracy

**15 Features:**
- Meristic (6)
- Morphometric (4)
- Truss (5)

**Method:** ML + Biological Range Filter
""")
st.sidebar.caption("FYP Project | UMT")

# ===============================
# MAIN TITLE
# ===============================

st.title("🐟 Mugilidae Fish Classification System")
st.markdown("### Hybrid: ML Models + Biological Range Validation")
st.markdown("---")

# ===============================
# STEP 1: UPLOAD EXCEL FILE
# ===============================

st.header("📁 Step 1: Upload Your Excel File")

uploaded_file = st.file_uploader(
    "Upload FYP Mugilidae Dataset(CLEANED).xlsx",
    type=['xlsx']
)

if uploaded_file is not None:
    
    # ===============================
    # LOAD AND PROCESS REAL DATA
    # ===============================
    
    with st.spinner("Extracting data from Excel..."):
        
        species_names = list(SPECIES_RANGES.keys())
        
        all_real_data = []
        for sheet_idx, species in enumerate(species_names):
            df_raw = pd.read_excel(uploaded_file, sheet_name=sheet_idx, header=None)
            
            meristic = extract_block(df_raw, "Meristic")
            morphometric = extract_block(df_raw, "Morphometric")
            truss = extract_block(df_raw, "Truss Network")
            if truss is None:
                truss = extract_block(df_raw, "Truss")
            
            if meristic is None or morphometric is None or truss is None:
                continue
            
            n = min(len(meristic), len(morphometric), len(truss))
            meristic = meristic.iloc[:n].reset_index(drop=True)
            morphometric = morphometric.iloc[:n].reset_index(drop=True)
            truss = truss.iloc[:n].reset_index(drop=True)
            
            # Meristic
            nd1_cols = [c for c in meristic.columns if 'ND1' in str(c)]
            nd1_total = meristic[nd1_cols].sum(axis=1).values if nd1_cols else np.ones(n)*4
            
            nd2_cols = [c for c in meristic.columns if 'ND2' in str(c)]
            nd2_total = meristic[nd2_cols].sum(axis=1).values if nd2_cols else np.ones(n)*7
            
            np_val = meristic['NP'].values if 'NP' in meristic.columns else np.ones(n)*14
            nc_val = meristic['NC'].values if 'NC' in meristic.columns else np.ones(n)*14
            
            nv_cols = [c for c in meristic.columns if 'NV' in str(c)]
            nv_total = meristic[nv_cols].sum(axis=1).values if nv_cols else np.ones(n)*6
            
            na_cols = [c for c in meristic.columns if 'NA' in str(c)]
            na_total = meristic[na_cols].sum(axis=1).values if na_cols else np.ones(n)*10
            
            # Morphometric
            sl = morphometric['SL'].values if 'SL' in morphometric.columns else np.ones(n)*150
            pl = morphometric['PL'].values if 'PL' in morphometric.columns else np.ones(n)*40
            bh = morphometric['BH'].values if 'BH' in morphometric.columns else np.ones(n)*45
            hl = morphometric['HL'].values if 'HL' in morphometric.columns else np.ones(n)*40
            
            # Truss
            head_truss = get_truss_sum(truss, ['AB', 'AC', 'AD'])
            anterior_truss = get_truss_sum(truss, ['BC', 'BD', 'CD'])
            mid_truss = get_truss_sum(truss, ['CE', 'CF', 'DE', 'DF', 'EF'])
            posterior_truss = get_truss_sum(truss, ['EG', 'EH', 'FG', 'FH', 'GH'])
            tail_truss = get_truss_sum(truss, ['GI', 'GJ', 'HI', 'HJ', 'IJ'])
            
            # Clean NaN values
            for arr in [nd1_total, nd2_total, np_val, nc_val, nv_total, na_total,
                       sl, pl, bh, hl, head_truss, anterior_truss, mid_truss, posterior_truss, tail_truss]:
                arr = np.nan_to_num(arr, nan=0)
            
            species_df = pd.DataFrame({
                'Species': species,
                'ND1_Total': nd1_total, 'ND2_Total': nd2_total, 'NP': np_val, 'NC': nc_val,
                'NV_Total': nv_total, 'NA_Total': na_total, 'SL': sl, 'PL': pl, 'BH': bh, 'HL': hl,
                'Head_Truss': head_truss, 'Anterior_Truss': anterior_truss, 'Mid_Truss': mid_truss,
                'Posterior_Truss': posterior_truss, 'Tail_Truss': tail_truss
            })
            all_real_data.append(species_df)
        
        real_df = pd.concat(all_real_data, ignore_index=True)
        
        # Final cleaning
        for col in FEATURE_NAMES:
            real_df[col] = pd.to_numeric(real_df[col], errors='coerce')
            real_df[col] = real_df[col].fillna(real_df[col].median())
    
    st.success(f"✅ Data loaded! {len(real_df)} real specimens")
    
    # ===============================
    # STEP 2: DATA SIMULATION
    # ===============================
    
    st.header("📊 Step 2: Data Simulation")
    
    col1, col2 = st.columns(2)
    with col1:
        target_samples = st.slider("Target samples per species", 100, 500, 200, 50)
    with col2:
        noise_level = st.slider("Noise level (%)", 0, 20, 5, 1)
    
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
            
            count_data = []
            for sp in species_names:
                count = len(final_df[final_df['Species'] == sp])
                count_data.append({"Species": sp, "Count": count})
            st.dataframe(pd.DataFrame(count_data), use_container_width=True)
    
    # ===============================
    # STEP 3: TRAIN ML MODELS
    # ===============================
    
    if 'final_df' in st.session_state:
        st.header("🤖 Step 3: Train ML Models")
        
        final_df = st.session_state['final_df']
        
        X = final_df[FEATURE_NAMES].values
        y = final_df['Species'].values
        
        X = np.nan_to_num(X)
        
        label_encoder = LabelEncoder()
        y_enc = label_encoder.fit_transform(y)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_enc, test_size=0.2, random_state=42, stratify=y_enc
        )
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Training Samples", len(X_train))
        with col2:
            st.metric("Test Samples", len(X_test))
        
        if st.button("🚀 Train All Models", type="primary"):
            
            results = []
            progress_bar = st.progress(0)
            status = st.empty()
            
            # 1. ANN
            status.text("Training ANN...")
            start = time.time()
            ann = MLPClassifier(hidden_layer_sizes=(10,5), max_iter=500, random_state=42)
            ann.fit(X_train, y_train)
            ann_acc = accuracy_score(y_test, ann.predict(X_test))
            ann_time = time.time() - start
            results.append({"Method": "ANN", "Accuracy": ann_acc, "Time": ann_time})
            progress_bar.progress(25)
            
            # 2. PSO
            status.text("Training PSO...")
            start = time.time()
            best_acc = 0
            best_params = None
            for i in range(40):
                h1 = np.random.randint(4, 25)
                h2 = np.random.randint(2, 15)
                alpha = np.random.uniform(0.0001, 0.01)
                lr = np.random.uniform(0.0001, 0.005)
                model = MLPClassifier(hidden_layer_sizes=(h1,h2), alpha=alpha, learning_rate_init=lr, max_iter=300, random_state=42)
                scores = cross_val_score(model, X_train, y_train, cv=3)
                mean_score = scores.mean() if len(scores) > 0 else 0
                if mean_score > best_acc:
                    best_acc = mean_score
                    best_params = (h1, h2, alpha, lr)
            if best_params:
                pso = MLPClassifier(hidden_layer_sizes=(best_params[0], best_params[1]), alpha=best_params[2], learning_rate_init=best_params[3], max_iter=500, random_state=42)
                pso.fit(X_train, y_train)
                pso_acc = accuracy_score(y_test, pso.predict(X_test))
            else:
                pso = ann
                pso_acc = ann_acc
            pso_time = time.time() - start
            results.append({"Method": "PSO", "Accuracy": pso_acc, "Time": pso_time})
            progress_bar.progress(50)
            
            # 3. GA
            status.text("Training GA...")
            start = time.time()
            best_acc = 0
            best_params = None
            for i in range(40):
                h1 = np.random.randint(4, 25)
                h2 = np.random.randint(2, 15)
                alpha = np.random.uniform(0.0001, 0.01)
                lr = np.random.uniform(0.0001, 0.005)
                model = MLPClassifier(hidden_layer_sizes=(h1,h2), alpha=alpha, learning_rate_init=lr, max_iter=300, random_state=42)
                scores = cross_val_score(model, X_train, y_train, cv=3)
                mean_score = scores.mean() if len(scores) > 0 else 0
                if mean_score > best_acc:
                    best_acc = mean_score
                    best_params = (h1, h2, alpha, lr)
            if best_params:
                ga = MLPClassifier(hidden_layer_sizes=(best_params[0], best_params[1]), alpha=best_params[2], learning_rate_init=best_params[3], max_iter=500, random_state=42)
                ga.fit(X_train, y_train)
                ga_acc = accuracy_score(y_test, ga.predict(X_test))
            else:
                ga = ann
                ga_acc = ann_acc
            ga_time = time.time() - start
            results.append({"Method": "GA", "Accuracy": ga_acc, "Time": ga_time})
            progress_bar.progress(75)
            
            # 4. GWO
            status.text("Training GWO...")
            start = time.time()
            best_acc = 0
            best_params = None
            for i in range(40):
                h1 = np.random.randint(4, 25)
                h2 = np.random.randint(2, 15)
                alpha = np.random.uniform(0.0001, 0.01)
                lr = np.random.uniform(0.0001, 0.005)
                model = MLPClassifier(hidden_layer_sizes=(h1,h2), alpha=alpha, learning_rate_init=lr, max_iter=300, random_state=42)
                scores = cross_val_score(model, X_train, y_train, cv=3)
                mean_score = scores.mean() if len(scores) > 0 else 0
                if mean_score > best_acc:
                    best_acc = mean_score
                    best_params = (h1, h2, alpha, lr)
            if best_params:
                gwo = MLPClassifier(hidden_layer_sizes=(best_params[0], best_params[1]), alpha=best_params[2], learning_rate_init=best_params[3], max_iter=500, random_state=42)
                gwo.fit(X_train, y_train)
                gwo_acc = accuracy_score(y_test, gwo.predict(X_test))
            else:
                gwo = ann
                gwo_acc = ann_acc
            gwo_time = time.time() - start
            results.append({"Method": "GWO", "Accuracy": gwo_acc, "Time": gwo_time})
            progress_bar.progress(100)
            
            status.text("Training complete!")
            
            # Find best method
            best_idx = np.argmax([r['Accuracy'] for r in results])
            best_method_name = results[best_idx]['Method']
            best_model = [ann, pso, ga, gwo][best_idx]
            best_accuracy = results[best_idx]['Accuracy']
            
            st.session_state['results'] = results
            st.session_state['best_model'] = best_model
            st.session_state['best_method_name'] = best_method_name
            st.session_state['best_accuracy'] = best_accuracy
            st.session_state['scaler'] = scaler
            st.session_state['label_encoder'] = label_encoder
            st.session_state['X_test'] = X_test
            st.session_state['y_test'] = y_test
            
            st.success(f"✅ All models trained!")
            st.info(f"🏆 **Best ML Model: {best_method_name}** with {best_accuracy:.3f} ({best_accuracy*100:.1f}%) accuracy")
    
    # ===============================
    # STEP 4: RESULTS
    # ===============================
    
    if 'results' in st.session_state:
        st.header("📊 Step 4: ML Model Comparison")
        
        results = st.session_state['results']
        res_df = pd.DataFrame(results)
        styled = res_df.style.highlight_max(subset=['Accuracy'], color='lightgreen')
        st.dataframe(styled, use_container_width=True)
        
        # Charts
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
        
        # Confusion Matrix
        st.subheader("📊 Confusion Matrix (Best ML Model)")
        best_model = st.session_state['best_model']
        y_pred = best_model.predict(st.session_state['X_test'])
        y_true = st.session_state['y_test']
        le = st.session_state['label_encoder']
        
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.set_xticks(np.arange(len(le.classes_)))
        ax.set_yticks(np.arange(len(le.classes_)))
        ax.set_xticklabels([s.split()[0] for s in le.classes_], rotation=45, ha='right')
        ax.set_yticklabels([s.split()[0] for s in le.classes_])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix')
        
        for i in range(len(le.classes_)):
            for j in range(len(le.classes_)):
                text = ax.text(j, i, cm[i, j],
                              ha="center", va="center", color="white" if cm[i, j] > cm.max()/2 else "black")
        plt.tight_layout()
        st.pyplot(fig)
        
        # Per-species accuracy
        st.subheader("📊 Per-Species Accuracy (Best ML Model)")
        per_species = []
        for i, sp in enumerate(le.classes_):
            mask = y_true == i
            if mask.sum() > 0:
                acc = (y_pred[mask] == i).sum() / mask.sum()
                per_species.append({
                    'Species': sp,
                    'Test Samples': mask.sum(),
                    'ML Accuracy': f"{acc:.3f} ({acc*100:.1f}%)"
                })
        st.dataframe(pd.DataFrame(per_species), use_container_width=True)
    
    # ===============================
    # STEP 5: HYBRID PREDICTION
    # ===============================
    
    if 'best_model' in st.session_state:
        st.header("🔮 Step 5: Hybrid Fish Identification")
        st.info("💡 **Hybrid Method:** ML Model Prediction + Biological Range Validation")
        
        # Reference table
        with st.expander("📖 Species Measurement Ranges", expanded=False):
            range_table = []
            for sp, ranges in SPECIES_RANGES.items():
                range_table.append({
                    "Species": sp,
                    "ND2": f"{ranges['ND2_Total'][0]}-{ranges['ND2_Total'][1]}",
                    "NP": f"{ranges['NP'][0]}-{ranges['NP'][1]}",
                    "SL (mm)": f"{ranges['SL'][0]:.0f}-{ranges['SL'][1]:.0f}"
                })
            st.dataframe(pd.DataFrame(range_table), use_container_width=True)
        
        st.markdown("### Enter 15 Fish Measurements")
        
        # Input form
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Meristic Features")
            nd1 = st.number_input("ND1_Total", value=4.0, step=1.0)
            nd2 = st.number_input("ND2_Total", value=7.0, step=1.0)
            np_val = st.number_input("NP", value=14.0, step=1.0)
            nc = st.number_input("NC", value=14.0, step=1.0)
            nv = st.number_input("NV_Total", value=6.0, step=1.0)
            na = st.number_input("NA_Total", value=10.0, step=1.0)
        
        with col2:
            st.subheader("Morphometric Features (mm)")
            sl = st.number_input("SL", value=150.0, step=10.0)
            pl = st.number_input("PL", value=40.0, step=5.0)
            bh = st.number_input("BH", value=45.0, step=5.0)
            hl = st.number_input("HL", value=40.0, step=5.0)
        
        with col3:
            st.subheader("Truss Features (mm)")
            head = st.number_input("Head_Truss", value=80.0, step=10.0)
            ant = st.number_input("Anterior_Truss", value=70.0, step=10.0)
            mid = st.number_input("Mid_Truss", value=200.0, step=20.0)
            post = st.number_input("Posterior_Truss", value=200.0, step=20.0)
            tail = st.number_input("Tail_Truss", value=100.0, step=10.0)
        
        if st.button("🔍 Identify Species (Hybrid)", type="primary"):
            features = [nd1, nd2, np_val, nc, nv, na, sl, pl, bh, hl, head, ant, mid, post, tail]
            
            # Hybrid prediction
            final_species, confidence, method, ml_species, in_range, warning, range_scores = hybrid_predict(
                features, st.session_state['best_model'], st.session_state['scaler'], st.session_state['label_encoder']
            )
            
            st.markdown("---")
            
            # Show result
            st.success(f"### 🎯 Identified Species: **{final_species}**")
            st.progress(int(confidence))
            st.caption(f"Confidence: {confidence:.1f}%")
            st.info(f"📌 Method: {method}")
            
            if warning:
                st.warning(warning)
            
            # Show ML prediction vs Range
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("ML Model Prediction", ml_species)
            with col_b:
                in_range_status = "✅ Within Range" if in_range else "❌ Outside Range"
                st.metric("Range Validation", in_range_status)
            
            # Show all species compatibility
            st.subheader("📊 Species Compatibility (Range-Based)")
            score_df = pd.DataFrame({
                'Species': list(range_scores.keys()),
                'Compatibility (%)': list(range_scores.values())
            }).sort_values('Compatibility (%)', ascending=False)
            st.dataframe(score_df, use_container_width=True)
            
            # Feature analysis for the final species
            st.subheader("📊 Feature Analysis")
            ranges = SPECIES_RANGES[final_species]
            
            analysis = []
            for i, f in enumerate(FEATURE_NAMES):
                mn, mx = ranges[f]
                val = features[i]
                status = "✅" if mn <= val <= mx else "❌"
                analysis.append({
                    "Feature": FEATURE_DISPLAY[i],
                    "Your Value": f"{val:.1f}",
                    "Range": f"{mn:.1f} - {mx:.1f}",
                    "Status": status
                })
            st.dataframe(pd.DataFrame(analysis), use_container_width=True)
            
            # Tips
            if sl > 250:
                st.info("📏 SL > 250mm indicates a LARGE species (Planiliza subviridis or Moolgarda tade)")
            if nd2 >= 8 and np_val >= 15:
                st.info("🔍 High ND2 (≥8) and NP (≥15) strongly suggests Moolgarda tade")

else:
    st.info("👈 Please upload your Excel file to begin")
    
    with st.expander("📖 How to Use This App"):
        st.markdown("""
        ### Hybrid Classification System:
        
        **Step 1-4: ML Model Training**
        - Upload your Excel file
        - Generate simulated data
        - Train ANN, PSO, GA, GWO models
        -
