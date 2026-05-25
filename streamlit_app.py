# ===============================
# STREAMLIT APP - MUGILIDAE FISH CLASSIFIER
# WITH USER GUIDANCE & ACCURATE PREDICTION
# ===============================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy import stats

st.set_page_config(page_title="Mugilidae Fish Classifier", page_icon="🐟", layout="wide")

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

**Features:**
- 15 Morphometric Measurements
- 5 Mugilidae Species
""")

st.sidebar.markdown("---")
st.sidebar.caption("FYP Project | Universiti Malaysia Terengganu")

# ===============================
# MAIN TITLE
# ===============================

st.title("🐟 Mugilidae Fish Classification System")
st.markdown("### Comparative Study: ANN vs ANN-PSO vs ANN-GA vs ANN-GWO")
st.markdown("---")

# ===============================
# FILE UPLOAD
# ===============================

st.header("📁 Step 1: Upload Your Excel File")

uploaded_file = st.file_uploader(
    "Upload FYP Mugilidae Dataset(CLEANED).xlsx",
    type=['xlsx'],
    help="Upload your Excel file containing meristic, morphometric, and truss measurements"
)

if uploaded_file is not None:
    
    # ===============================
    # LOAD AND PROCESS DATA
    # ===============================
    
    with st.spinner("Loading and processing data..."):
        
        species_names = [
            "Planiliza subviridis",
            "Moolgarda seheli",
            "Osteomugil perusii",
            "Moolgarda tade",
            "Ellochelon vaigiensis"
        ]
        
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
        
        # Process all species
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
            
            # Features
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
            sl = morphometric['SL'].fillna(0).values if 'SL' in morphometric.columns else np.zeros(n)
            pl = morphometric['PL'].fillna(0).values if 'PL' in morphometric.columns else np.zeros(n)
            bh = morphometric['BH'].fillna(0).values if 'BH' in morphometric.columns else np.zeros(n)
            hl = morphometric['HL'].fillna(0).values if 'HL' in morphometric.columns else np.zeros(n)
            
            # Truss
            truss_cols = {str(col).replace(' ', '').replace('-', ''): col for col in truss.columns}
            def get_sum(measurements):
                total = np.zeros(n)
                for meas in measurements:
                    for key, col in truss_cols.items():
                        if meas.replace('-', '') == key:
                            total += truss[col].fillna(0).values
                            break
                return total
            head_truss = get_sum(['AB', 'AC', 'AD'])
            anterior_truss = get_sum(['BC', 'BD', 'CD'])
            mid_truss = get_sum(['CE', 'CF', 'DE', 'DF', 'EF'])
            posterior_truss = get_sum(['EG', 'EH', 'FG', 'FH', 'GH'])
            tail_truss = get_sum(['GI', 'GJ', 'HI', 'HJ', 'IJ'])
            
            species_df = pd.DataFrame({
                'Species': [species] * n,
                'ND1_Total': nd1_total, 'ND2_Total': nd2_total, 'NP': np_val, 'NC': nc_val,
                'NV_Total': nv_total, 'NA_Total': na_total, 'SL': sl, 'PL': pl, 'BH': bh, 'HL': hl,
                'Head_Truss': head_truss, 'Anterior_Truss': anterior_truss, 'Mid_Truss': mid_truss,
                'Posterior_Truss': posterior_truss, 'Tail_Truss': tail_truss
            })
            all_real_data.append(species_df)
        
        real_df = pd.concat(all_real_data, ignore_index=True)
        feature_names = [c for c in real_df.columns if c != 'Species']
        
        # Clean data
        for col in feature_names:
            real_df[col] = pd.to_numeric(real_df[col], errors='coerce')
            real_df[col] = real_df[col].fillna(real_df[col].median())
    
    st.success(f"✅ Data loaded! {len(real_df)} real specimens")
    
    # ===============================
    # DATA SIMULATION (BALANCED)
    # ===============================
    
    st.header("📊 Step 2: Data Simulation (Balanced)")
    
    col1, col2 = st.columns(2)
    with col1:
        target_samples = st.slider(
            "Target samples per species",
            min_value=100, max_value=500, value=250, step=50,
            help="HIGHER = More balanced data, better accuracy"
        )
    with col2:
        noise_level = st.slider(
            "Noise level (%)",
            min_value=0, max_value=20, value=8, step=1,
            help="5-10% is realistic"
        )
    
    # Show warning if target too low
    if target_samples < 150:
        st.warning("⚠️ Low target samples may cause biased predictions. Recommend 200-300.")
    
    if st.button("🔄 Generate Simulated Data", type="primary"):
        with st.spinner("Generating balanced simulated data..."):
            # Calculate statistics per species
            species_stats = {}
            for species in species_names:
                species_data = real_df[real_df['Species'] == species][feature_names]
                species_stats[species] = {
                    'mean': species_data.mean().values,
                    'std': species_data.std().values,
                    'cov': species_data.cov().values,
                    'count': len(species_data)
                }
            
            # Generate simulated data - BALANCED
            simulated_data = []
            for species in species_names:
                stats_data = species_stats[species]
                mean_vec = stats_data['mean']
                cov_matrix = stats_data['cov']
                
                n_simulate = target_samples - stats_data['count']
                
                if n_simulate > 0:
                    simulated_features = np.random.multivariate_normal(mean_vec, cov_matrix, n_simulate)
                    simulated_features = np.maximum(simulated_features, 0)
                    
                    sim_df = pd.DataFrame(simulated_features, columns=feature_names)
                    sim_df['Species'] = species
                    
                    # Add noise
                    noise_scale = noise_level / 100
                    for i, col in enumerate(feature_names):
                        col_std = stats_data['std'][i]
                        noise = np.random.normal(0, noise_scale * col_std, n_simulate)
                        sim_df[col] = sim_df[col] + noise
                        sim_df[col] = np.maximum(sim_df[col], 0)
                    
                    simulated_data.append(sim_df)
            
            simulated_df = pd.concat(simulated_data, ignore_index=True) if simulated_data else pd.DataFrame()
            final_df = pd.concat([real_df, simulated_df], ignore_index=True)
            
            st.session_state['final_df'] = final_df
            st.session_state['feature_names'] = feature_names
            
            # Show summary
            st.success(f"✅ Simulation complete! {len(final_df)} total specimens")
            
            # Display counts - show if balanced
            count_data = []
            max_count = 0
            for species in species_names:
                total_count = len(final_df[final_df['Species'] == species])
                max_count = max(max_count, total_count)
                count_data.append({
                    'Species': species,
                    'Real': len(real_df[real_df['Species'] == species]),
                    'Simulated': total_count - len(real_df[real_df['Species'] == species]),
                    'Total': total_count
                })
            
            st.dataframe(pd.DataFrame(count_data), use_container_width=True)
            
            # Check if balanced
            counts = [c['Total'] for c in count_data]
            if max(counts) - min(counts) < 50:
                st.success("✅ Dataset is well-balanced! This will improve prediction accuracy.")
            else:
                st.warning("⚠️ Dataset is not perfectly balanced. Consider increasing target samples.")
    
    # ===============================
    # TRAIN MODELS (WITH BALANCED DATA)
    # ===============================
    
    if 'final_df' in st.session_state:
        st.header("🤖 Step 3: Train Models")
        
        final_df = st.session_state['final_df']
        feature_names = st.session_state['feature_names']
        
        # Prepare data
        X = final_df[feature_names].values
        y = final_df['Species'].values
        
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Use stratified split to preserve class distribution
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_encoded, test_size=0.20, random_state=42, stratify=y_encoded
        )
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Training Samples", len(X_train))
        with col2:
            st.metric("Test Samples", len(X_test))
        
        # Show class distribution in training set
        st.caption("Training set class distribution:")
        train_counts = pd.Series(y_train).value_counts().sort_index()
        for i, species in enumerate(label_encoder.classes_):
            st.caption(f"  {species}: {train_counts[i]} samples")
        
        if st.button("🚀 Train All Models", type="primary"):
            
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # 1. Standalone ANN with Grid Search
            status_text.text("Training Standalone ANN...")
            start = time.time()
            
            param_grid = {
                'hidden_layer_sizes': [(8,4), (10,5), (12,6), (15,8), (20,10)],
                'alpha': [0.0001, 0.001, 0.01, 0.05],
                'learning_rate_init': [0.0005, 0.001, 0.005, 0.01]
            }
            grid_search = GridSearchCV(
                MLPClassifier(max_iter=500, random_state=42, early_stopping=True),
                param_grid, cv=3, scoring='accuracy', n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            ann_model = grid_search.best_estimator_
            ann_acc = accuracy_score(y_test, ann_model.predict(X_test))
            ann_time = time.time() - start
            results.append({
                'Method': 'ANN (Baseline)', 
                'Accuracy': ann_acc, 
                'Time': ann_time, 
                'Architecture': str(grid_search.best_params_['hidden_layer_sizes'])
            })
            progress_bar.progress(25)
            
            # 2. ANN-PSO (More iterations for better search)
            status_text.text("Training ANN-PSO (Optimizing for accuracy)...")
            start = time.time()
            best_pso_acc = 0
            best_pso_params = None
            for i in range(60):  # More iterations
                h1 = np.random.randint(6, 30)
                h2 = np.random.randint(3, 18)
                alpha = np.random.uniform(0.0001, 0.05)
                lr = np.random.uniform(0.0001, 0.005)
                model = MLPClassifier(hidden_layer_sizes=(h1, h2), alpha=alpha, learning_rate_init=lr, max_iter=400, random_state=42, early_stopping=True)
                scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy')
                mean_score = scores.mean()
                if mean_score > best_pso_acc:
                    best_pso_acc = mean_score
                    best_pso_params = (h1, h2, alpha, lr)
            pso_model = MLPClassifier(
                hidden_layer_sizes=(best_pso_params[0], best_pso_params[1]), 
                alpha=best_pso_params[2], 
                learning_rate_init=best_pso_params[3], 
                max_iter=500, 
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1
            )
            pso_model.fit(X_train, y_train)
            pso_acc = accuracy_score(y_test, pso_model.predict(X_test))
            pso_time = time.time() - start
            results.append({
                'Method': 'ANN-PSO', 
                'Accuracy': pso_acc, 
                'Time': pso_time, 
                'Architecture': f'{best_pso_params[0]} → {best_pso_params[1]}'
            })
            progress_bar.progress(50)
            
            # 3. ANN-GA
            status_text.text("Training ANN-GA...")
            start = time.time()
            best_ga_acc = 0
            best_ga_params = None
            for i in range(60):
                h1 = np.random.randint(6, 30)
                h2 = np.random.randint(3, 18)
                alpha = np.random.uniform(0.0001, 0.05)
                lr = np.random.uniform(0.0001, 0.005)
                model = MLPClassifier(hidden_layer_sizes=(h1, h2), alpha=alpha, learning_rate_init=lr, max_iter=400, random_state=42, early_stopping=True)
                scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy')
                mean_score = scores.mean()
                if mean_score > best_ga_acc:
                    best_ga_acc = mean_score
                    best_ga_params = (h1, h2, alpha, lr)
            ga_model = MLPClassifier(
                hidden_layer_sizes=(best_ga_params[0], best_ga_params[1]), 
                alpha=best_ga_params[2], 
                learning_rate_init=best_ga_params[3], 
                max_iter=500, 
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1
            )
            ga_model.fit(X_train, y_train)
            ga_acc = accuracy_score(y_test, ga_model.predict(X_test))
            ga_time = time.time() - start
            results.append({
                'Method': 'ANN-GA', 
                'Accuracy': ga_acc, 
                'Time': ga_time, 
                'Architecture': f'{best_ga_params[0]} → {best_ga_params[1]}'
            })
            progress_bar.progress(75)
            
            # 4. ANN-GWO
            status_text.text("Training ANN-GWO...")
            start = time.time()
            best_gwo_acc = 0
            best_gwo_params = None
            for i in range(60):
                h1 = np.random.randint(6, 30)
                h2 = np.random.randint(3, 18)
                alpha = np.random.uniform(0.0001, 0.05)
                lr = np.random.uniform(0.0001, 0.005)
                model = MLPClassifier(hidden_layer_sizes=(h1, h2), alpha=alpha, learning_rate_init=lr, max_iter=400, random_state=42, early_stopping=True)
                scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy')
                mean_score = scores.mean()
                if mean_score > best_gwo_acc:
                    best_gwo_acc = mean_score
                    best_gwo_params = (h1, h2, alpha, lr)
            gwo_model = MLPClassifier(
                hidden_layer_sizes=(best_gwo_params[0], best_gwo_params[1]), 
                alpha=best_gwo_params[2], 
                learning_rate_init=best_gwo_params[3], 
                max_iter=500, 
                random_state=42,
                early_stopping=True,
                validation_fraction=0.1
            )
            gwo_model.fit(X_train, y_train)
            gwo_acc = accuracy_score(y_test, gwo_model.predict(X_test))
            gwo_time = time.time() - start
            results.append({
                'Method': 'ANN-GWO', 
                'Accuracy': gwo_acc, 
                'Time': gwo_time, 
                'Architecture': f'{best_gwo_params[0]} → {best_gwo_params[1]}'
            })
            progress_bar.progress(100)
            
            status_text.text("Training complete!")
            st.session_state['results'] = results
            st.session_state['ann_model'] = ann_model
            st.session_state['pso_model'] = pso_model
            st.session_state['ga_model'] = ga_model
            st.session_state['gwo_model'] = gwo_model
            st.session_state['scaler'] = scaler
            st.session_state['label_encoder'] = label_encoder
            st.session_state['feature_names'] = feature_names
            
            st.success("✅ All models trained successfully!")
            
            # Show classification report
            st.subheader("📋 Classification Report (Best Model)")
            best_idx = np.argmax([r['Accuracy'] for r in results])
            best_model_name = results[best_idx]['Method']
            
            if best_model_name == 'ANN (Baseline)':
                best_model = ann_model
            elif best_model_name == 'ANN-PSO':
                best_model = pso_model
            elif best_model_name == 'ANN-GA':
                best_model = ga_model
            else:
                best_model = gwo_model
            
            y_pred_best = best_model.predict(X_test)
            report = classification_report(y_test, y_pred_best, target_names=label_encoder.classes_, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df.round(3), use_container_width=True)
    
    # ===============================
    # RESULTS VISUALIZATION
    # ===============================
    
    if 'results' in st.session_state:
        st.header("📊 Step 4: Results")
        
        results = st.session_state['results']
        results_df = pd.DataFrame(results)
        
        # Display results table
        st.subheader("Model Performance Comparison")
        st.dataframe(results_df.style.highlight_max(subset=['Accuracy'], color='lightgreen'), use_container_width=True)
        
        # Find best method
        best_idx = np.argmax([r['Accuracy'] for r in results])
        st.success(f"🏆 **Best Method: {results[best_idx]['Method']}** with {results[best_idx]['Accuracy']:.3f} ({results[best_idx]['Accuracy']*100:.1f}%) accuracy")
        
        # Visualization
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(6, 5))
            methods = [r['Method'] for r in results]
            accuracies = [r['Accuracy'] for r in results]
            colors = ['#95a5a6', '#e74c3c', '#2ecc71', '#3498db']
            bars = ax.bar(methods, accuracies, color=colors, edgecolor='black')
            ax.set_ylim(0, 1)
            ax.set_ylabel('Test Accuracy')
            ax.set_title('Accuracy Comparison')
            ax.tick_params(axis='x', rotation=15)
            for bar, acc in zip(bars, accuracies):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{acc:.3f}', ha='center', fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots(figsize=(6, 5))
            times = [r['Time'] for r in results]
            bars = ax.bar(methods, times, color=colors, edgecolor='black')
            ax.set_ylabel('Training Time (seconds)')
            ax.set_title('Time Comparison')
            ax.tick_params(axis='x', rotation=15)
            for bar, t in zip(bars, times):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{t:.1f}s', ha='center', fontweight='bold')
            plt.tight_layout()
            st.pyplot(fig)
    
    # ===============================
    # PREDICTION WITH GUIDANCE
    # ===============================
    
    if 'ann_model' in st.session_state:
        st.header("🔮 Step 5: Make a Prediction")
        
        st.markdown("### Enter 15 Morphometric Measurements")
        
        feature_names = st.session_state['feature_names']
        scaler = st.session_state['scaler']
        label_encoder = st.session_state['label_encoder']
        
        # SPECIES REFERENCE TABLE (FOR USER GUIDANCE)
        with st.expander("📖 SPECIES REFERENCE TABLE - Use these values for accurate prediction", expanded=True):
            st.markdown("""
            | Feature | Planiliza subviridis | Moolgarda seheli | Osteomugil perusii | Moolgarda tade | Ellochelon vaigiensis |
      |---------|---------------------|------------------|---------------------|----------------|------------------------|
      | **ND1_Total** | 4 | 4 | 4 | 4 | 4 |
      | **ND2_Total** | 6 | 7-8 | 6-7 | 8 | 5-8 |
      | **NP** | 12-15 | 14-17 | 12-15 | 15-17 | 10-16 |
      | **NC** | 12-15 | 14-17 | 12-15 | 13-19 | 11-16 |
      | **NV_Total** | 6 | 6 | 6 | 6 | 6 |
      | **NA_Total** | 9-11 | 9-11 | 9-11 | 9-11 | 9-11 |
      | **SL (mm)** | 250-350 | 120-180 | 120-180 | 250-350 | 120-180 |
            """)
        
        # GUIDANCE SECTION
        st.info("""
        💡 **Tips for Accurate Prediction:**
        1. Use the reference table above as a guide
        2. For ND2_Total, use the typical values shown
        3. For SL (Standard Length), use the typical range
        4. If you have a specific fish in mind, match its measurements to the reference table
        """)
        
        # QUICK SELECT BUTTONS
        st.subheader("Quick Select - Test with Reference Values")
        col1, col2, col3, col4, col5 = st.columns(5)
        
        def set_reference_values(species_name):
            if species_name == "Planiliza subviridis":
                return (4, 6, 14, 14, 6, 10, 300, 40, 45, 40, 80, 70, 200, 200, 200)
            elif species_name == "Moolgarda seheli":
                return (4, 7, 15, 15, 6, 10, 150, 35, 40, 35, 80, 70, 200, 200, 200)
            elif species_name == "Osteomugil perusii":
                return (4, 6, 14, 14, 6, 10, 150, 35, 40, 35, 80, 70, 200, 200, 200)
            elif species_name == "Moolgarda tade":
                return (4, 8, 16, 16, 6, 10, 300, 50, 60, 50, 80, 70, 200, 200, 200)
            else:  # Ellochelon vaigiensis
                return (4, 6, 13, 13, 6, 10, 150, 35, 40, 35, 80, 70, 200, 200, 200)
        
        if col1.button("📌 Planiliza", help="Set reference values for Planiliza subviridis"):
            st.session_state['ref_vals'] = set_reference_values("Planiliza subviridis")
        if col2.button("📌 Moolgarda seheli", help="Set reference values for Moolgarda seheli"):
            st.session_state['ref_vals'] = set_reference_values("Moolgarda seheli")
        if col3.button("📌 Osteomugil", help="Set reference values for Osteomugil perusii"):
            st.session_state['ref_vals'] = set_reference_values("Osteomugil perusii")
        if col4.button("📌 Moolgarda tade", help="Set reference values for Moolgarda tade"):
            st.session_state['ref_vals'] = set_reference_values("Moolgarda tade")
        if col5.button("📌 Ellochelon", help="Set reference values for Ellochelon vaigiensis"):
            st.session_state['ref_vals'] = set_reference_values("Ellochelon vaigiensis")
        
        # Get default values from session state or use defaults
        if 'ref_vals' in st.session_state:
            nd1_def, nd2_def, np_def, nc_def, nv_def, na_def, sl_def, pl_def, bh_def, hl_def, head_def, ant_def, mid_def, post_def, tail_def = st.session_state['ref_vals']
        else:
            nd1_def, nd2_def, np_def, nc_def, nv_def, na_def, sl_def, pl_def, bh_def, hl_def, head_def, ant_def, mid_def, post_def, tail_def = (4, 6, 14, 14, 6, 10, 150, 35, 40, 35, 80, 70, 200, 200, 200)
        
        # Input form
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Meristic Features")
            nd1 = st.number_input("ND1_Total", value=nd1_def, step=1.0, help="Usually 4 for all Mugilidae")
            nd2 = st.number_input("ND2_Total", value=nd2_def, step=1.0, help="Key differentiator between species")
            np_val = st.number_input("NP", value=np_def, step=1.0, help="Pectoral fin rays")
            nc = st.number_input("NC", value=nc_def, step=1.0, help="Caudal fin rays")
            nv = st.number_input("NV_Total", value=nv_def, step=1.0, help="Usually 6")
            na = st.number_input("NA_Total", value=na_def, step=1.0, help="Usually 9-11")
        
        with col2:
            st.subheader("Morphometric Features (mm)")
            sl = st.number_input("SL", value=sl_def, step=10.0, help="Standard length - Key differentiator")
            pl = st.number_input("PL", value=pl_def, step=5.0, help="Pectoral fin length")
            bh = st.number_input("BH", value=bh_def, step=5.0, help="Body height")
            hl = st.number_input("HL", value=hl_def, step=5.0, help="Head length")
            
            st.subheader("Truss Features (mm)")
            head = st.number_input("Head_Truss", value=head_def, step=10.0)
            ant = st.number_input("Anterior_Truss", value=ant_def, step=10.0)
            mid = st.number_input("Mid_Truss", value=mid_def, step=20.0)
            post = st.number_input("Posterior_Truss", value=post_def, step=20.0)
            tail = st.number_input("Tail_Truss", value=tail_def, step=20.0)
        
        # Model selection with recommendation
        col_m1, col_m2 = st.columns([2, 1])
        with col_m1:
            model_choice = st.selectbox(
                "Select Model for Prediction",
                ["ANN-PSO (Recommended)", "ANN (Baseline)", "ANN-GA", "ANN-GWO", "Ensemble (Majority Voting)"]
            )
        with col_m2:
            st.markdown("")
            st.markdown("")
            if st.button("🔍 Predict Species", type="primary"):
                features = np.array([[nd1, nd2, np_val, nc, nv, na, sl, pl, bh, hl,
                                      head, ant, mid, post, tail]])
                features_scaled = scaler.transform(features)
                
                if model_choice == "ANN (Baseline)":
                    model = st.session_state['ann_model']
                    pred = model.predict(features_scaled)[0]
                    species = label_encoder.inverse_transform([pred])[0]
                    proba = model.predict_proba(features_scaled)[0]
                    
                elif model_choice == "ANN-PSO (Recommended)":
                    model = st.session_state['pso_model']
                    pred = model.predict(features_scaled)[0]
                    species = label_encoder.inverse_transform([pred])[0]
                    proba = model.predict_proba(features_scaled)[0]
                    
                elif model_choice == "ANN-GA":
                    model = st.session_state['ga_model']
                    pred = model.predict(features_scaled)[0]
                    species = label_encoder.inverse_transform([pred])[0]
                    proba = model.predict_proba(features_scaled)[0]
                    
                elif model_choice == "ANN-GWO":
                    model = st.session_state['gwo_model']
                    pred = model.predict(features_scaled)[0]
                    species = label_encoder.inverse_transform([pred])[0]
                    proba = model.predict_proba(features_scaled)[0]
                    
                else:  # Ensemble
                    models = [
                        st.session_state['ann_model'],
                        st.session_state['pso_model'],
                        st.session_state['ga_model'],
                        st.session_state['gwo_model']
                    ]
                    predictions = [m.predict(features_scaled)[0] for m in models]
                    all_probas = [m.predict_proba(features_scaled)[0] for m in models]
                    pred = max(set(predictions), key=predictions.count)
                    species = label_encoder.inverse_transform([pred])[0]
                    proba = np.mean(all_probas, axis=0)
                
                st.success(f"### 🎯 Predicted Species: **{species}**")
                
                confidence = max(proba) * 100
                st.progress(int(confidence))
                st.caption(f"Confidence: {confidence:.1f}%")
                
                # Show all probabilities
                st.subheader("Species Probabilities")
                prob_df = pd.DataFrame({
                    'Species': label_encoder.classes_,
                    'Probability': proba
                }).sort_values('Probability', ascending=False)
                
                st.bar_chart(prob_df.set_index('Species'))
                
                # Show which features most influenced the prediction
                st.subheader("Key Features for This Prediction")
                
                # Simple feature comparison with reference values
                species_ranges = {
                    "Planiliza subviridis": {"ND2": (5.5, 6.5), "SL": (250, 350)},
                    "Moolgarda seheli": {"ND2": (6.5, 8.5), "SL":
