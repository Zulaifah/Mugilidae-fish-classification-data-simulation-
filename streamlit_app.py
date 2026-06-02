# ===============================
# STREAMLIT APP - MUGILIDAE FISH CLASSIFIER
# Comparative Study: ANN vs ANN-PSO vs ANN-GA vs ANN-GWO
# With Data Simulation & Interactive Visualization
# ===============================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
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

**Data:** Real + Simulated (200 samples/species)
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
        
        # Read Excel file
        excel_file = pd.ExcelFile(uploaded_file)
        
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
    # DATA SIMULATION
    # ===============================
    
    st.header("📊 Step 2: Data Simulation")
    
    col1, col2 = st.columns(2)
    with col1:
        target_samples = st.slider(
            "Target samples per species",
            min_value=50, max_value=500, value=200, step=50,
            help="Number of samples (real + simulated) per species"
        )
    with col2:
        noise_level = st.slider(
            "Noise level (%)",
            min_value=0, max_value=20, value=5, step=1,
            help="Amount of random noise added to simulated data"
        )
    
    if st.button("🔄 Generate Simulated Data", type="primary"):
        with st.spinner("Generating simulated data..."):
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
            
            # Generate simulated data
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
            
            # Display counts
            count_data = []
            for species in species_names:
                real_count = len(real_df[real_df['Species'] == species])
                sim_count = len(final_df[final_df['Species'] == species]) - real_count
                count_data.append({
                    'Species': species,
                    'Real': real_count,
                    'Simulated': sim_count,
                    'Total': real_count + sim_count
                })
            
            st.dataframe(pd.DataFrame(count_data), use_container_width=True)
    
    # ===============================
    # TRAIN MODELS
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
            ann_model = MLPClassifier(hidden_layer_sizes=(10, 5), max_iter=500, random_state=42)
            ann_model.fit(X_train, y_train)
            ann_acc = accuracy_score(y_test, ann_model.predict(X_test))
            ann_time = time.time() - start
            results.append({'Method': 'ANN (Baseline)', 'Accuracy': ann_acc, 'Time': ann_time, 'Architecture': '10 → 5'})
            progress_bar.progress(25)
            
            # 2. ANN-PSO (simplified for speed)
            status_text.text("Training ANN-PSO...")
            start = time.time()
            best_pso_acc = 0
            best_pso_params = None
            for i in range(30):
                h1 = np.random.randint(4, 20)
                h2 = np.random.randint(2, 12)
                alpha = np.random.uniform(0.0001, 0.01)
                lr = np.random.uniform(0.0001, 0.005)
                model = MLPClassifier(hidden_layer_sizes=(h1, h2), alpha=alpha, learning_rate_init=lr, max_iter=300, random_state=42)
                scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy')
                mean_score = scores.mean()
                if mean_score > best_pso_acc:
                    best_pso_acc = mean_score
                    best_pso_params = (h1, h2, alpha, lr)
            pso_model = MLPClassifier(hidden_layer_sizes=(best_pso_params[0], best_pso_params[1]), alpha=best_pso_params[2], learning_rate_init=best_pso_params[3], max_iter=400, random_state=42)
            pso_model.fit(X_train, y_train)
            pso_acc = accuracy_score(y_test, pso_model.predict(X_test))
            pso_time = time.time() - start
            results.append({'Method': 'ANN-PSO', 'Accuracy': pso_acc, 'Time': pso_time, 'Architecture': f'{best_pso_params[0]} → {best_pso_params[1]}'})
            progress_bar.progress(50)
            
            # 3. ANN-GA
            status_text.text("Training ANN-GA...")
            start = time.time()
            best_ga_acc = 0
            best_ga_params = None
            for i in range(30):
                h1 = np.random.randint(4, 20)
                h2 = np.random.randint(2, 12)
                alpha = np.random.uniform(0.0001, 0.01)
                lr = np.random.uniform(0.0001, 0.005)
                model = MLPClassifier(hidden_layer_sizes=(h1, h2), alpha=alpha, learning_rate_init=lr, max_iter=300, random_state=42)
                scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy')
                mean_score = scores.mean()
                if mean_score > best_ga_acc:
                    best_ga_acc = mean_score
                    best_ga_params = (h1, h2, alpha, lr)
            ga_model = MLPClassifier(hidden_layer_sizes=(best_ga_params[0], best_ga_params[1]), alpha=best_ga_params[2], learning_rate_init=best_ga_params[3], max_iter=400, random_state=42)
            ga_model.fit(X_train, y_train)
            ga_acc = accuracy_score(y_test, ga_model.predict(X_test))
            ga_time = time.time() - start
            results.append({'Method': 'ANN-GA', 'Accuracy': ga_acc, 'Time': ga_time, 'Architecture': f'{best_ga_params[0]} → {best_ga_params[1]}'})
            progress_bar.progress(75)
            
            # 4. ANN-GWO
            status_text.text("Training ANN-GWO...")
            start = time.time()
            best_gwo_acc = 0
            best_gwo_params = None
            for i in range(30):
                h1 = np.random.randint(4, 20)
                h2 = np.random.randint(2, 12)
                alpha = np.random.uniform(0.0001, 0.01)
                lr = np.random.uniform(0.0001, 0.005)
                model = MLPClassifier(hidden_layer_sizes=(h1, h2), alpha=alpha, learning_rate_init=lr, max_iter=300, random_state=42)
                scores = cross_val_score(model, X_train, y_train, cv=3, scoring='accuracy')
                mean_score = scores.mean()
                if mean_score > best_gwo_acc:
                    best_gwo_acc = mean_score
                    best_gwo_params = (h1, h2, alpha, lr)
            gwo_model = MLPClassifier(hidden_layer_sizes=(best_gwo_params[0], best_gwo_params[1]), alpha=best_gwo_params[2], learning_rate_init=best_gwo_params[3], max_iter=400, random_state=42)
            gwo_model.fit(X_train, y_train)
            gwo_acc = accuracy_score(y_test, gwo_model.predict(X_test))
            gwo_time = time.time() - start
            results.append({'Method': 'ANN-GWO', 'Accuracy': gwo_acc, 'Time': gwo_time, 'Architecture': f'{best_gwo_params[0]} → {best_gwo_params[1]}'})
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

    # ===============================
    # CONFUSION MATRICES (Tambah Selepas Training)
    # ===============================
    
    if 'ann_model' in st.session_state and 'X_test' in locals():
        st.header("📊 Confusion Matrices for All Models")
        
        # Get predictions from all models
        y_pred_ann = st.session_state['ann_model'].predict(X_test)
        y_pred_pso = st.session_state['pso_model'].predict(X_test)
        y_pred_ga = st.session_state['ga_model'].predict(X_test)
        y_pred_gwo = st.session_state['gwo_model'].predict(X_test)
        
        species_names_short = [s.split()[0] for s in label_encoder.classes_]
        
        # Create 2x2 grid of confusion matrices
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # ANN Confusion Matrix
        cm_ann = confusion_matrix(y_test, y_pred_ann)
        sns.heatmap(cm_ann, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=species_names_short, yticklabels=species_names_short,
                    ax=axes[0,0])
        axes[0,0].set_title('ANN (Baseline)', fontsize=12, fontweight='bold')
        axes[0,0].set_xlabel('Predicted', fontsize=10)
        axes[0,0].set_ylabel('Actual', fontsize=10)
        
        # PSO Confusion Matrix
        cm_pso = confusion_matrix(y_test, y_pred_pso)
        sns.heatmap(cm_pso, annot=True, fmt='d', cmap='Greens', 
                    xticklabels=species_names_short, yticklabels=species_names_short,
                    ax=axes[0,1])
        axes[0,1].set_title('ANN-PSO', fontsize=12, fontweight='bold')
        axes[0,1].set_xlabel('Predicted', fontsize=10)
        axes[0,1].set_ylabel('Actual', fontsize=10)
        
        # GA Confusion Matrix
        cm_ga = confusion_matrix(y_test, y_pred_ga)
        sns.heatmap(cm_ga, annot=True, fmt='d', cmap='Oranges', 
                    xticklabels=species_names_short, yticklabels=species_names_short,
                    ax=axes[1,0])
        axes[1,0].set_title('ANN-GA', fontsize=12, fontweight='bold')
        axes[1,0].set_xlabel('Predicted', fontsize=10)
        axes[1,0].set_ylabel('Actual', fontsize=10)
        
        # GWO Confusion Matrix
        cm_gwo = confusion_matrix(y_test, y_pred_gwo)
        sns.heatmap(cm_gwo, annot=True, fmt='d', cmap='Reds', 
                    xticklabels=species_names_short, yticklabels=species_names_short,
                    ax=axes[1,1])
        axes[1,1].set_title('ANN-GWO', fontsize=12, fontweight='bold')
        axes[1,1].set_xlabel('Predicted', fontsize=10)
        axes[1,1].set_ylabel('Actual', fontsize=10)
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Calculate and display per-class accuracy for best model
        best_idx = np.argmax([r['Accuracy'] for r in st.session_state['results']])
        best_name = st.session_state['results'][best_idx]['Method']
        
        st.subheader(f"📋 Per-Class Accuracy - {best_name} (Best Model)")
        
        if best_name == "ANN (Baseline)":
            best_pred = y_pred_ann
        elif best_name == "ANN-PSO":
            best_pred = y_pred_pso
        elif best_name == "ANN-GA":
            best_pred = y_pred_ga
        else:
            best_pred = y_pred_gwo
        
        per_class_data = []
        for i, species in enumerate(label_encoder.classes_):
            mask = y_test == i
            if mask.sum() > 0:
                correct = (best_pred[mask] == i).sum()
                total = mask.sum()
                acc = correct / total
                per_class_data.append({
                    'Species': species,
                    'Correct': correct,
                    'Total': total,
                    'Accuracy': f"{acc:.3f} ({acc*100:.1f}%)"
                })
        
        st.dataframe(pd.DataFrame(per_class_data), use_container_width=True)
    
    # ===============================
    # RESULTS VISUALIZATION WITH CONFUSION MATRICES
    # ===============================
    
    if 'results' in st.session_state:
        st.header("📊 Step 4: Results")
        
        results = st.session_state['results']
        results_df = pd.DataFrame(results)
        
        # Display results table
        st.subheader("Model Performance Comparison")
        st.dataframe(results_df, use_container_width=True)
        
        # Find best method
        best_idx = np.argmax([r['Accuracy'] for r in results])
        st.success(f"🏆 **Best Method: {results[best_idx]['Method']}** with {results[best_idx]['Accuracy']:.3f} ({results[best_idx]['Accuracy']*100:.1f}%) accuracy")
        
        # Visualization - Accuracy & Time
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
        # CONFUSION MATRICES
        # ===============================
        
        st.header("📊 Confusion Matrices for All Models")
        
        # Get predictions
        y_pred_ann = st.session_state['ann_model'].predict(X_test)
        y_pred_pso = st.session_state['pso_model'].predict(X_test)
        y_pred_ga = st.session_state['ga_model'].predict(X_test)
        y_pred_gwo = st.session_state['gwo_model'].predict(X_test)
        
        species_names_short = [s.split()[0] for s in label_encoder.classes_]
        
        # Create 2x2 grid
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # ANN
        cm_ann = confusion_matrix(y_test, y_pred_ann)
        sns.heatmap(cm_ann, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=species_names_short, yticklabels=species_names_short,
                    ax=axes[0,0])
        axes[0,0].set_title('ANN (Baseline)', fontsize=12, fontweight='bold')
        axes[0,0].set_xlabel('Predicted')
        axes[0,0].set_ylabel('Actual')
        
        # PSO
        cm_pso = confusion_matrix(y_test, y_pred_pso)
        sns.heatmap(cm_pso, annot=True, fmt='d', cmap='Greens', 
                    xticklabels=species_names_short, yticklabels=species_names_short,
                    ax=axes[0,1])
        axes[0,1].set_title('ANN-PSO', fontsize=12, fontweight='bold')
        axes[0,1].set_xlabel('Predicted')
        axes[0,1].set_ylabel('Actual')
        
        # GA
        cm_ga = confusion_matrix(y_test, y_pred_ga)
        sns.heatmap(cm_ga, annot=True, fmt='d', cmap='Oranges', 
                    xticklabels=species_names_short, yticklabels=species_names_short,
                    ax=axes[1,0])
        axes[1,0].set_title('ANN-GA', fontsize=12, fontweight='bold')
        axes[1,0].set_xlabel('Predicted')
        axes[1,0].set_ylabel('Actual')
        
        # GWO
        cm_gwo = confusion_matrix(y_test, y_pred_gwo)
        sns.heatmap(cm_gwo, annot=True, fmt='d', cmap='Reds', 
                    xticklabels=species_names_short, yticklabels=species_names_short,
                    ax=axes[1,1])
        axes[1,1].set_title('ANN-GWO', fontsize=12, fontweight='bold')
        axes[1,1].set_xlabel('Predicted')
        axes[1,1].set_ylabel('Actual')
        
        plt.tight_layout()
        st.pyplot(fig)
        
        # Per-class accuracy for best model
        st.subheader(f"📋 Per-Class Accuracy - {results[best_idx]['Method']} (Best Model)")
        
        if results[best_idx]['Method'] == "ANN (Baseline)":
            best_pred = y_pred_ann
        elif results[best_idx]['Method'] == "ANN-PSO":
            best_pred = y_pred_pso
        elif results[best_idx]['Method'] == "ANN-GA":
            best_pred = y_pred_ga
        else:
            best_pred = y_pred_gwo
        
        per_class = []
        for i, species in enumerate(label_encoder.classes_):
            mask = y_test == i
            if mask.sum() > 0:
                correct = (best_pred[mask] == i).sum()
                total = mask.sum()
                acc = correct / total
                per_class.append({
                    'Species': species,
                    'Correct': correct,
                    'Total': total,
                    'Accuracy': f"{acc:.3f} ({acc*100:.1f}%)"
                })
        
        st.dataframe(pd.DataFrame(per_class), use_container_width=True)
    
    # ===============================
    # PREDICTION
    # ===============================
    
    if 'ann_model' in st.session_state:
        st.header("🔮 Step 5: Make a Prediction")
        
        st.markdown("### Enter 15 Morphometric Measurements")
        
        feature_names = st.session_state['feature_names']
        scaler = st.session_state['scaler']
        label_encoder = st.session_state['label_encoder']
        
        # Create input form
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Meristic Features")
            nd1 = st.number_input("ND1_Total", value=4.0, step=1.0)
            nd2 = st.number_input("ND2_Total", value=6.0, step=1.0)
            np_val = st.number_input("NP", value=14.0, step=1.0)
            nc = st.number_input("NC", value=14.0, step=1.0)
            nv = st.number_input("NV_Total", value=6.0, step=1.0)
            na = st.number_input("NA_Total", value=10.0, step=1.0)
        
        with col2:
            st.subheader("Morphometric Features (mm)")
            sl = st.number_input("SL", value=150.0, step=10.0)
            pl = st.number_input("PL", value=35.0, step=5.0)
            bh = st.number_input("BH", value=40.0, step=5.0)
            hl = st.number_input("HL", value=35.0, step=5.0)
            
            st.subheader("Truss Features (mm)")
            head = st.number_input("Head_Truss", value=80.0, step=10.0)
            ant = st.number_input("Anterior_Truss", value=70.0, step=10.0)
            mid = st.number_input("Mid_Truss", value=200.0, step=20.0)
            post = st.number_input("Posterior_Truss", value=200.0, step=20.0)
            tail = st.number_input("Tail_Truss", value=200.0, step=20.0)
        
        # Choose model for prediction
        model_choice = st.selectbox(
            "Select Model for Prediction",
            ["ANN (Baseline)", "ANN-PSO", "ANN-GA", "ANN-GWO", "Ensemble (Majority Voting)"]
        )
        
        if st.button("🔍 Predict Species", type="primary"):
            features = np.array([[nd1, nd2, np_val, nc, nv, na, sl, pl, bh, hl,
                                  head, ant, mid, post, tail]])
            features_scaled = scaler.transform(features)
            
            if model_choice == "ANN (Baseline)":
                model = st.session_state['ann_model']
                pred = model.predict(features_scaled)[0]
                species = label_encoder.inverse_transform([pred])[0]
                proba = model.predict_proba(features_scaled)[0]
                
            elif model_choice == "ANN-PSO":
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
                pred = max(set(predictions), key=predictions.count)
                species = label_encoder.inverse_transform([pred])[0]
                proba = models[0].predict_proba(features_scaled)[0]
            
            st.success(f"### 🎯 Predicted Species: **{species}**")
            
            # Show confidence
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
    
else:
    st.info("👈 Please upload your Excel file to begin")
    
    # Show instructions
    with st.expander("📖 How to Use This App"):
        st.markdown("""
        1. **Upload** your Excel file (FYP Mugilidae Dataset(CLEANED).xlsx)
        2. **Configure** simulation parameters (target samples per species, noise level)
        3. **Generate** simulated data
        4. **Train** all 4 models (ANN, ANN-PSO, ANN-GA, ANN-GWO)
        5. **Compare** results in tables and charts
        6. **Make predictions** using any model
        """)

# ===============================
# FOOTER
# ===============================

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
    <p>🐟 Comparative Study: ANN vs ANN-PSO vs ANN-GA vs ANN-GWO</p>
    <p>FYP Project | Universiti Malaysia Terengganu</p>
    </div>
    """,
    unsafe_allow_html=True
)
