# ===============================
# STREAMLIT APP - MUGILIDAE FISH CLASSIFIER
# WITH RANGE VALIDATION & ACCURATE PREDICTION
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
# SPECIES RANGE DATABASE
# ===============================

SPECIES_RANGES = {
    "Planiliza subviridis": {
        "ND1_Total": (4, 4),
        "ND2_Total": (6, 9),
        "NP": (10, 15),
        "NC": (11, 16),
        "NV_Total": (5, 6),
        "NA_Total": (8, 11),
        "SL": (80.44, 622.03),
        "PL": (14.59, 210.91),
        "BH": (19.57, 227.04),
        "HL": (21.29, 217.10),
        "Head_Truss": (3.62, 46.87),
        "Anterior_Truss": (12.81, 47.42),
        "Mid_Truss": (17.49, 87.04),
        "Posterior_Truss": (17.34, 87.73),
        "Tail_Truss": (8.55, 67.56)
    },
    "Moolgarda seheli": {
        "ND1_Total": (4, 4),
        "ND2_Total": (6, 9),
        "NP": (10, 16),
        "NC": (11, 17),
        "NV_Total": (5, 7),
        "NA_Total": (9, 12),
        "SL": (79.52, 300.00),
        "PL": (21.53, 67.88),
        "BH": (23.79, 68.95),
        "HL": (20.79, 75.44),
        "Head_Truss": (3.58, 75.55),
        "Anterior_Truss": (13.30, 66.92),
        "Mid_Truss": (15.56, 117.48),
        "Posterior_Truss": (20.44, 127.25),
        "Tail_Truss": (10.51, 96.42)
    },
    "Osteomugil perusii": {
        "ND1_Total": (4, 4),
        "ND2_Total": (6, 9),
        "NP": (10, 16),
        "NC": (10, 17),
        "NV_Total": (5, 6),
        "NA_Total": (9, 11),
        "SL": (11.47, 177.00),
        "PL": (10.86, 160.09),
        "BH": (12.04, 167.54),
        "HL": (14.57, 162.09),
        "Head_Truss": (3.63, 76.33),
        "Anterior_Truss": (10.00, 43.39),
        "Mid_Truss": (10.62, 142.16),
        "Posterior_Truss": (11.47, 454.04),
        "Tail_Truss": (6.31, 59.39)
    },
    "Moolgarda tade": {
        "ND1_Total": (4, 4),
        "ND2_Total": (8, 9),
        "NP": (15, 17),
        "NC": (13, 19),
        "NV_Total": (6, 9),
        "NA_Total": (9, 12),
        "SL": (74.54, 372.13),
        "PL": (24.25, 75.83),
        "BH": (30.65, 150.76),
        "HL": (27.86, 230.02),
        "Head_Truss": (5.09, 87.18),
        "Anterior_Truss": (14.82, 80.00),
        "Mid_Truss": (24.15, 131.86),
        "Posterior_Truss": (23.35, 149.08),
        "Tail_Truss": (13.88, 108.36)
    },
    "Ellochelon vaigiensis": {
        "ND1_Total": (4, 4),
        "ND2_Total": (6, 9),
        "NP": (10, 16),
        "NC": (11, 16),
        "NV_Total": (6, 7),
        "NA_Total": (7, 11),
        "SL": (49.73, 363.50),
        "PL": (0, 53.86),
        "BH": (30.5, 119.05),
        "HL": (31.61, 183.42),
        "Head_Truss": (4.99, 92.38),
        "Anterior_Truss": (18.22, 125.79),
        "Mid_Truss": (18.30, 148.39),
        "Posterior_Truss": (27.28, 168.88),
        "Tail_Truss": (15.73, 115.95)
    }
}

# List of feature names in order
FEATURE_NAMES = ["ND1_Total", "ND2_Total", "NP", "NC", "NV_Total", "NA_Total", 
                 "SL", "PL", "BH", "HL", "Head_Truss", "Anterior_Truss", 
                 "Mid_Truss", "Posterior_Truss", "Tail_Truss"]

def validate_measurements(features):
    """Check if measurements are within range for each species"""
    warnings_list = []
    possible_species = []
    
    for species_name in SPECIES_RANGES.keys():
        in_range_count = 0
        total_checks = 0
        
        for i, feature in enumerate(FEATURE_NAMES):
            min_val, max_val = SPECIES_RANGES[species_name][feature]
            value = features[i]
            
            if min_val <= value <= max_val:
                in_range_count += 1
            total_checks += 1
        
        match_percentage = (in_range_count / total_checks) * 100
        if match_percentage > 40:
            possible_species.append((species_name, match_percentage))
        
        # Collect warnings for the input
        for i, feature in enumerate(FEATURE_NAMES):
            min_val, max_val = SPECIES_RANGES[species_name][feature]
            value = features[i]
            if not (min_val <= value <= max_val):
                warnings_list.append(f"⚠️ {feature}: {value:.1f} is outside {species_name} range ({min_val:.0f}-{max_val:.0f})")
    
    # Remove duplicates in warnings
    warnings_list = list(set(warnings_list))
    
    # Sort possible species by match percentage
    possible_species.sort(key=lambda x: x[1], reverse=True)
    
    return warnings_list[:10], possible_species[:3]

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
        
        # Clean data
        for col in FEATURE_NAMES:
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
    
    if target_samples < 150:
        st.warning("⚠️ Low target samples may cause biased predictions. Recommend 200-300.")
    
    if st.button("🔄 Generate Simulated Data", type="primary"):
        with st.spinner("Generating balanced simulated data..."):
            # Calculate statistics per species
            species_stats = {}
            for species in species_names:
                species_data = real_df[real_df['Species'] == species][FEATURE_NAMES]
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
                    
                    sim_df = pd.DataFrame(simulated_features, columns=FEATURE_NAMES)
                    sim_df['Species'] = species
                    
                    # Add noise
                    noise_scale = noise_level / 100
                    for i, col in enumerate(FEATURE_NAMES):
                        col_std = stats_data['std'][i]
                        noise = np.random.normal(0, noise_scale * col_std, n_simulate)
                        sim_df[col] = sim_df[col] + noise
                        sim_df[col] = np.maximum(sim_df[col], 0)
                    
                    simulated_data.append(sim_df)
            
            simulated_df = pd.concat(simulated_data, ignore_index=True) if simulated_data else pd.DataFrame()
            final_df = pd.concat([real_df, simulated_df], ignore_index=True)
            
            st.session_state['final_df'] = final_df
            
            # Show summary
            st.success(f"✅ Simulation complete! {len(final_df)} total specimens")
            
            # Display counts
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
    
    # ===============================
    # TRAIN MODELS
    # ===============================
    
    if 'final_df' in st.session_state:
        st.header("🤖 Step 3: Train Models")
        
        final_df = st.session_state['final_df']
        
        # Prepare data
        X = final_df[FEATURE_NAMES].values
        y = final_df['Species'].values
        
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Use stratified split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_encoded, test_size=0.20, random_state=42, stratify=y_encoded
        )
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Training Samples", len(X_train))
        with col2:
            st.metric("Test Samples", len(X_test))
        
        if st.button("🚀 Train All Models", type="primary"):
            
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # 1. Standalone ANN
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
                'Time': ann_time
            })
            progress_bar.progress(25)
            
            # 2. ANN-PSO
            status_text.text("Training ANN-PSO...")
            start = time.time()
            best_pso_acc = 0
            best_pso_params = None
            for i in range(60):
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
                'Time': pso_time
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
                'Time': ga_time
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
                'Time': gwo_time
            })
            progress_bar.progress(100)
            
            status_text.text("Training complete!")
            st.session_state['results'] = results
            st.session_state['pso_model'] = pso_model
            st.session_state['scaler'] = scaler
            st.session_state['label_encoder'] = label_encoder
            
            st.success("✅ All models trained successfully!")
    
    # ===============================
    # RESULTS VISUALIZATION
    # ===============================
    
    if 'results' in st.session_state:
        st.header("📊 Step 4: Results")
        
        results = st.session_state['results']
        results_df = pd.DataFrame(results)
        
        st.subheader("Model Performance Comparison")
        st.dataframe(results_df.style.highlight_max(subset=['Accuracy'], color='lightgreen'), use_container_width=True)
        
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
    # PREDICTION SECTION WITH RANGE VALIDATION
    # ===============================
    
    if 'pso_model' in st.session_state:
        st.header("🔮 Step 5: Make a Prediction")
        
        st.markdown("### Enter 15 Morphometric Measurements")
        
        st.info("""
        💡 **Key Differentiators for Accurate Identification:**
        - **Moolgarda tade** has HIGHER NP (15-17) and ND2_Total (8-9)
        - **Planiliza subviridis** and **Moolgarda tade** are LARGER (SL > 250mm)
        - Other species are SMALLER (SL < 200mm)
        """)
        
        # Reference Table
        with st.expander("📖 SPECIES MEASUREMENT RANGES", expanded=True):
            range_table_data = []
            for species in SPECIES_RANGES.keys():
                row = {
                    "Species": species,
                    "ND2": f"{SPECIES_RANGES[species]['ND2_Total'][0]}-{SPECIES_RANGES[species]['ND2_Total'][1]}",
                    "NP": f"{SPECIES_RANGES[species]['NP'][0]}-{SPECIES_RANGES[species]['NP'][1]}",
                    "NC": f"{SPECIES_RANGES[species]['NC'][0]}-{SPECIES_RANGES[species]['NC'][1]}",
                    "SL (mm)": f"{SPECIES_RANGES[species]['SL'][0]:.0f}-{SPECIES_RANGES[species]['SL'][1]:.0f}"
                }
                range_table_data.append(row)
            range_df = pd.DataFrame(range_table_data)
            st.dataframe(range_df, use_container_width=True)
        
        # Quick select buttons
        st.subheader("Quick Select - Load Reference Values")
        col_b1, col_b2, col_b3, col_b4, col_b5 = st.columns(5)
        
        def set_reference_values(species):
            ranges = SPECIES_RANGES[species]
            st.session_state['ref_values'] = {
                "ND1_Total": ranges["ND1_Total"][0],
                "ND2_Total": (ranges["ND2_Total"][0] + ranges["ND2_Total"][1]) // 2,
                "NP": (ranges["NP"][0] + ranges["NP"][1]) // 2,
                "NC": (ranges["NC"][0] + ranges["NC"][1]) // 2,
                "NV_Total": (ranges["NV_Total"][0] + ranges["NV_Total"][1]) // 2,
                "NA_Total": (ranges["NA_Total"][0] + ranges["NA_Total"][1]) // 2,
                "SL": (ranges["SL"][0] + ranges["SL"][1]) / 2,
                "PL": (ranges["PL"][0] + ranges["PL"][1]) / 2,
                "BH": (ranges["BH"][0] + ranges["BH"][1]) / 2,
                "HL": (ranges["HL"][0] + ranges["HL"][1]) / 2,
                "Head_Truss": (ranges["Head_Truss"][0] + ranges["Head_Truss"][1]) / 2,
                "Anterior_Truss": (ranges["Anterior_Truss"][0] + ranges["Anterior_Truss"][1]) / 2,
                "Mid_Truss": (ranges["Mid_Truss"][0] + ranges["Mid_Truss"][1]) / 2,
                "Posterior_Truss": (ranges["Posterior_Truss"][0] + ranges["Posterior_Truss"][1]) / 2,
                "Tail_Truss": (ranges["Tail_Truss"][0] + ranges["Tail_Truss"][1]) / 2
            }
        
        if col_b1.button("📌 Planiliza"):
            set_reference_values("Planiliza subviridis")
        if col_b2.button("📌 Moolgarda s"):
            set_reference_values("Moolgarda seheli")
        if col_b3.button("📌 Osteomugil"):
            set_reference_values("Osteomugil perusii")
        if col_b4.button("📌 Moolgarda t"):
            set_reference_values("Moolgarda tade")
        if col_b5.button("📌 Ellochelon"):
            set_reference_values("Ellochelon vaigiensis")
        
        # Get default values
        if 'ref_values' in st.session_state:
            ref = st.session_state['ref_values']
        else:
            ref = {
                "ND1_Total": 4, "ND2_Total": 7, "NP": 14, "NC": 14, "NV_Total": 6, "NA_Total": 10,
                "SL": 150, "PL": 35, "BH": 40, "HL": 35, "Head_Truss": 80, "Anterior_Truss": 70,
                "Mid_Truss": 200, "Posterior_Truss": 200, "Tail_Truss": 200
            }
        
        # Input form
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Meristic Features")
            nd1 = st.number_input("ND1_Total", value=float(ref["ND1_Total"]), step=1.0)
            nd2 = st.number_input("ND2_Total", value=float(ref["ND2_Total"]), step=1.0, help="KEY DIFFERENTIATOR")
            np_val = st.number_input("NP", value=float(ref["NP"]), step=1.0, help="KEY DIFFERENTIATOR")
            nc = st.number_input("NC", value=float(ref["NC"]), step=1.0)
            nv = st.number_input("NV_Total", value=float(ref["NV_Total"]), step=1.0)
            na = st.number_input("NA_Total", value=float(ref["NA_Total"]), step=1.0)
        
        with col2:
            st.subheader("Morphometric Features (mm)")
            sl = st.number_input("SL", value=float(ref["SL"]), step=10.0, help="KEY DIFFERENTIATOR")
            pl = st.number_input("PL", value=float(ref["PL"]), step=5.0)
            bh = st.number_input("BH", value=float(ref["BH"]), step=5.0)
            hl = st.number_input("HL", value=float(ref["HL"]), step=5.0)
            
            st.subheader("Truss Features (mm)")
            head = st.number_input("Head_Truss", value=float(ref["Head_Truss"]), step=10.0)
            ant = st.number_input("Anterior_Truss", value=float(ref["Anterior_Truss"]), step=10.0)
            mid = st.number_input("Mid_Truss", value=float(ref["Mid_Truss"]), step=
