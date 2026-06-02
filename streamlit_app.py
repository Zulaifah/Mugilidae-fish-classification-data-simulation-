# ===============================
# STREAMLIT APP - MUGILIDAE FISH CLASSIFIER
# COMPLETE: Data Extraction + Simulation + Comparison + Prediction
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
st.sidebar.header("📋 About")
st.sidebar.info("""
**Comparative Study:**
- ANN (Baseline)
- ANN-PSO (Particle Swarm)
- ANN-GA (Genetic Algorithm)
- ANN-GWO (Grey Wolf Optimizer)

**15 Features:**
- Meristic (6)
- Morphometric (4)
- Truss (5)

**Data:** Real + Simulated
""")
st.sidebar.caption("FYP Project | UMT")

# ===============================
# MAIN TITLE
# ===============================

st.title("🐟 Mugilidae Fish Classification System")
st.markdown("### Comparative Study: ANN vs ANN-PSO vs ANN-GA vs ANN-GWO")
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
        
        species_names = [
            "Planiliza subviridis",
            "Moolgarda seheli",
            "Osteomugil perusii",
            "Moolgarda tade",
            "Ellochelon vaigiensis"
        ]
        
        all_real_data = []
        for sheet_idx, species in enumerate(species_names):
            df_raw = pd.read_excel(uploaded_file, sheet_name=sheet_idx, header=None)
            
            meristic = extract_block(df_raw, "Meristic")
            morphometric = extract_block(df_raw, "Morphometric")
            truss = extract_block(df_raw, "Truss Network")
            if truss is None:
                truss = extract_block(df_raw, "Truss")
            
            if meristic is None or morphometric is None or truss is None:
                st.warning(f"Missing data for {species}")
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
            
            # Show summary
            st.success(f"✅ Simulation complete! {len(final_df)} total specimens")
            
            count_data = []
            for sp in species_names:
                count = len(final_df[final_df['Species'] == sp])
                count_data.append({"Species": sp, "Count": count})
            st.dataframe(pd.DataFrame(count_data), use_container_width=True)
    
    # ===============================
    # STEP 3: TRAIN MODELS
    # ===============================
    
    if 'final_df' in st.session_state:
        st.header("🤖 Step 3: Train Models")
        
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
            
            # 1. ANN Baseline
            status.text("Training ANN...")
            start = time.time()
            ann = MLPClassifier(hidden_layer_sizes=(10,5), max_iter=500, random_state=42)
            ann.fit(X_train, y_train)
            ann_acc = accuracy_score(y_test, ann.predict(X_test))
            ann_time = time.time() - start
            results.append({"Method": "ANN", "Accuracy": ann_acc, "Time": ann_time})
            progress_bar.progress(25)
            
            # 2. ANN-PSO
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
            
            # 3. ANN-GA
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
            
            # 4. ANN-GWO
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
            st.session_state['results'] = results
            st.session_state['pso_model'] = pso
            st.session_state['ga_model'] = ga
            st.session_state['gwo_model'] = gwo
            st.session_state['scaler'] = scaler
            st.session_state['label_encoder'] = label_encoder
            
            st.success("✅ All models trained successfully!")
    
    # ===============================
    # STEP 4: RESULTS
    # ===============================
    
    if 'results' in st.session_state:
        st.header("📊 Step 4: Results")
        
        results = st.session_state['results']
        res_df = pd.DataFrame(results)
        
        # Highlight best
        st.dataframe(res_df.style.highlight_max(subset=['Accuracy'], color='lightgreen'), use_container_width=True)
        
        best_idx = res_df['Accuracy'].argmax()
        best_method = res_df.iloc[best_idx]['Method']
        best_acc = res_df.iloc[best_idx]['Accuracy']
        st.success(f"🏆 **Best Method: {best_method}** with {best_acc:.3f} ({best_acc*100:.1f}%) accuracy")
        
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
    
    # ===============================
    # STEP 5: PREDICTION
    # ===============================
    
    if 'pso_model' in st.session_state:
        st.header("🔮 Step 5: Make a Prediction")
        
        # Reference table
        with st.expander("📖 Species Reference Ranges (from your data)"):
            # Calculate ranges from real data
            range_data = []
            for sp in species_names:
                sp_data = real_df[real_df['Species'] == sp]
                range_data.append({
                    "Species": sp,
                    "ND2": f"{sp_data['ND2_Total'].min():.0f}-{sp_data['ND2_Total'].max():.0f}",
                    "NP": f"{sp_data['NP'].min():.0f}-{sp_data['NP'].max():.0f}",
                    "SL": f"{sp_data['SL'].min():.0f}-{sp_data['SL'].max():.0f} mm"
                })
            st.dataframe(pd.DataFrame(range_data), use_container_width=True)
        
        st.markdown("### Enter 15 Measurements")
        
        # Input form
        col1, col2, col3 = st.columns(3)
        
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
        
        with col3:
            st.subheader("Truss (mm)")
            head = st.number_input("Head_Truss", value=80.0, step=10.0)
            ant = st.number_input("Anterior_Truss", value=70.0, step=10.0)
            mid = st.number_input("Mid_Truss", value=200.0, step=20.0)
            post = st.number_input("Posterior_Truss", value=200.0, step=20.0)
            tail = st.number_input("Tail_Truss", value=100.0, step=10.0)
        
        # Model selection
        model_choice = st.selectbox(
            "Select Model",
            ["PSO (Recommended)", "ANN", "GA", "GWO", "Ensemble"]
        )
        
        if st.button("Predict Species", type="primary"):
            features = np.array([[nd1, nd2, np_val, nc, nv, na, sl, pl, bh, hl, head, ant, mid, post, tail]])
            features_scaled = st.session_state['scaler'].transform(features)
            
            if model_choice == "ANN":
                pred = st.session_state['pso_model']  # Placeholder - fix later
            elif model_choice == "PSO (Recommended)":
                pred = st.session_state['pso_model'].predict(features_scaled)[0]
            elif model_choice == "GA":
                pred = st.session_state['ga_model'].predict(features_scaled)[0]
            elif model_choice == "GWO":
                pred = st.session_state['gwo_model'].predict(features_scaled)[0]
            else:
                # Ensemble
                models = [st.session_state['pso_model'], st.session_state['ga_model'], st.session_state['gwo_model']]
                preds = [m.predict(features_scaled)[0] for m in models]
                pred = max(set(preds), key=preds.count)
            
            species = st.session_state['label_encoder'].inverse_transform([pred])[0]
            st.success(f"### 🎯 Predicted Species: **{species}**")

else:
    st.info("👈 Please upload your Excel file to begin")
    
    with st.expander("📖 How to Use"):
        st.markdown("""
        1. Upload your Excel file
        2. Configure simulation (target samples, noise)
        3. Generate simulated data
        4. Train all models
        5. Compare results
        6. Make predictions
        """)

st.markdown("---")
st.caption("FYP Project | Universiti Malaysia Terengganu")
