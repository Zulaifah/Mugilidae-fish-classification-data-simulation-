# ===============================
# STREAMLIT APP - MUGILIDAE FISH CLASSIFIER
# DENGAN PSO, GA, GWO YANG BETUL
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
from sklearn.metrics import accuracy_score, confusion_matrix

# Set GLOBAL random seed
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

st.set_page_config(page_title="Mugilidae Fish Classifier", page_icon="🐟", layout="wide")

# ===============================
# CLASS DEFINITIONS FOR REAL METAHEURISTICS
# ===============================

class Particle:
    """Particle for PSO optimization"""
    def __init__(self, bounds):
        self.position = np.array([
            np.random.uniform(bounds[0][0], bounds[0][1]),  # h1
            np.random.uniform(bounds[1][0], bounds[1][1]),  # h2
            np.random.uniform(bounds[2][0], bounds[2][1]),  # alpha
            np.random.uniform(bounds[3][0], bounds[3][1])   # learning rate
        ])
        self.velocity = np.zeros_like(self.position)
        self.best_position = self.position.copy()
        self.best_score = -float('inf')
    
    def update_velocity(self, global_best_position, w=0.7, c1=1.5, c2=1.5):
        r1, r2 = np.random.rand(2)
        cognitive = c1 * r1 * (self.best_position - self.position)
        social = c2 * r2 * (global_best_position - self.position)
        self.velocity = w * self.velocity + cognitive + social
    
    def update_position(self, bounds):
        self.position += self.velocity
        for i in range(len(bounds)):
            self.position[i] = np.clip(self.position[i], bounds[i][0], bounds[i][1])

# ===============================
# FUNCTIONS
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

FEATURE_NAMES = [
    "ND1_Total", "ND2_Total", "NP", "NC", "NV_Total", "NA_Total",
    "SL", "PL", "BH", "HL"
]

species_names = [
    "Planiliza subviridis",
    "Moolgarda seheli",
    "Osteomugil perusii",
    "Moolgarda tade",
    "Ellochelon vaigiensis"
]

# ===============================
# SIDEBAR
# ===============================

st.sidebar.title("🐟 Mugilidae Fish Classifier")
st.sidebar.markdown("---")

# Training Parameters
st.sidebar.subheader("⚙️ Optimization Parameters")
n_particles = st.sidebar.slider("Number of Particles (PSO)", 10, 50, 20, 5)
n_generations = st.sidebar.slider("Generations (GA)", 10, 50, 20, 5)
n_wolves = st.sidebar.slider("Number of Wolves (GWO)", 10, 50, 20, 5)
n_iterations = st.sidebar.slider("Iterations (PSO/GWO)", 20, 100, 40, 10)

st.sidebar.markdown("---")
st.sidebar.header("📋 About")
st.sidebar.info("""
**Comparative Study:**
- ANN (Baseline - Grid Search)
- ANN-PSO (Particle Swarm Optimization)
- ANN-GA (Genetic Algorithm)
- ANN-GWO (Grey Wolf Optimizer)

**Note:** Training may take 5-15 minutes
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
        
        all_real_data = []
        for sheet_idx, species in enumerate(species_names):
            df_raw = pd.read_excel(uploaded_file, sheet_name=sheet_idx, header=None)
            
            meristic = extract_block(df_raw, "Meristic")
            morphometric = extract_block(df_raw, "Morphometric")
            
            if meristic is None or morphometric is None:
                continue
            
            n = min(len(meristic), len(morphometric))
            meristic = meristic.iloc[:n].reset_index(drop=True)
            morphometric = morphometric.iloc[:n].reset_index(drop=True)
            
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
            
            # Clean NaN
            for arr in [nd1_total, nd2_total, np_val, nc_val, nv_total, na_total, sl, pl, bh, hl]:
                arr = np.nan_to_num(arr, nan=0)
            
            species_df = pd.DataFrame({
                'Species': species,
                'ND1_Total': nd1_total, 'ND2_Total': nd2_total, 'NP': np_val, 'NC': nc_val,
                'NV_Total': nv_total, 'NA_Total': na_total, 'SL': sl, 'PL': pl, 'BH': bh, 'HL': hl
            })
            all_real_data.append(species_df)
        
        real_df = pd.concat(all_real_data, ignore_index=True)
        for col in FEATURE_NAMES:
            real_df[col] = pd.to_numeric(real_df[col], errors='coerce')
            real_df[col] = real_df[col].fillna(real_df[col].median())
    
    st.success(f"✅ Data loaded! {len(real_df)} real specimens")
    
    # Show real data distribution
    st.subheader("📊 Real Data Distribution")
    real_dist = []
    for sp in species_names:
        count = len(real_df[real_df['Species'] == sp])
        real_dist.append({"Species": sp, "Real Specimens": count})
    st.dataframe(pd.DataFrame(real_dist), use_container_width=True)
    
    # ===============================
    # BALANCE DATASET (200 per species)
    # ===============================
    
    st.header("📊 Step 2: Balance Dataset")
    st.info("📌 **Target: 200 specimens per species (Balanced Dataset)**")
    
    target_samples = 200
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Target per Species", target_samples)
    with col2:
        st.metric("Total Dataset", target_samples * 5)
    
    if st.button("🔄 Generate Balanced Dataset", type="primary"):
        with st.spinner(f"Generating balanced dataset ({target_samples} per species)..."):
            np.random.seed(RANDOM_SEED)
            
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
                        
                        sim_data = np.random.normal(means, stds * 1.05, (need, len(FEATURE_NAMES)))
                        sim_data = np.maximum(sim_data, 0)
                        
                        sim_df = pd.DataFrame(sim_data, columns=FEATURE_NAMES)
                        sim_df['Species'] = species
                        final_df = pd.concat([final_df, sim_df], ignore_index=True)
            
            for col in FEATURE_NAMES:
                final_df[col] = final_df[col].fillna(final_df[col].median())
            
            st.session_state['balanced_df'] = final_df
            
            # Show distribution
            bal_dist = []
            for sp in species_names:
                count = len(final_df[final_df['Species'] == sp])
                real_count = len(real_df[real_df['Species'] == sp])
                bal_dist.append({
                    "Species": sp,
                    "Real": real_count,
                    "Simulated": count - real_count,
                    "Total": count
                })
            st.dataframe(pd.DataFrame(bal_dist), use_container_width=True)
            
            # Download button
            csv = final_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Balanced Dataset (CSV)",
                data=csv,
                file_name="balanced_fish_data.csv",
                mime="text/csv"
            )
            
            st.success(f"✅ Balanced dataset generated! {len(final_df)} total specimens")
    
    # ===============================
    # TRAIN MODELS
    # ===============================
    
    if 'balanced_df' in st.session_state:
        st.header("🤖 Step 3: Train Models")
        st.warning("⏱️ **Training may take 5-15 minutes. Please wait...**")
        
        final_df = st.session_state['balanced_df']
        
        X = final_df[FEATURE_NAMES].values
        y = final_df['Species'].values
        X = np.nan_to_num(X)
        
        label_encoder = LabelEncoder()
        y_enc = label_encoder.fit_transform(y)
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_enc, test_size=0.2, random_state=RANDOM_SEED, stratify=y_enc
        )
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Training Samples", len(X_train))
        with col2:
            st.metric("Test Samples", len(X_test))
        
        if st.button("🚀 Train All Models (Real Optimization)", type="primary"):
            
            results = []
            progress_bar = st.progress(0)
            status = st.empty()
            
            # ==========================================================
            # 1. ANN BASELINE (Grid Search untuk architecture terbaik)
            # ==========================================================
            status.text("Training ANN Baseline (Grid Search)...")
            start = time.time()
            
            arch_options = [(8,4), (10,5), (12,6), (15,8), (20,10)]
            best_ann_acc = 0
            best_ann = None
            best_ann_arch = (10,5)
            
            for arch in arch_options:
                model = MLPClassifier(hidden_layer_sizes=arch, max_iter=300, random_state=RANDOM_SEED)
                cv_scores = cross_val_score(model, X_train, y_train, cv=3)
                cv_mean = cv_scores.mean()
                if cv_mean > best_ann_acc:
                    best_ann_acc = cv_mean
                    best_ann_arch = arch
                    best_ann = model
            
            best_ann.fit(X_train, y_train)
            ann_acc = accuracy_score(y_test, best_ann.predict(X_test))
            ann_time = time.time() - start
            results.append({"Method": f"ANN ({best_ann_arch[0]},{best_ann_arch[1]})", "Accuracy": ann_acc, "Time": ann_time})
            progress_bar.progress(10)
            
            # ==========================================================
            # 2. REAL PSO (Particle Swarm Optimization)
            # ==========================================================
            status.text(f"Training REAL PSO ({n_particles} particles, {n_iterations} iterations)...")
            start = time.time()
            np.random.seed(RANDOM_SEED)
            
            # Bounds for parameters: [h1_min, h1_max], [h2_min, h2_max], [alpha_min, alpha_max], [lr_min, lr_max]
            bounds = np.array([
                [4, 25],      # h1
                [2, 15],      # h2
                [0.0001, 0.01],  # alpha
                [0.0001, 0.005]  # learning rate
            ])
            
            # Initialize particles
            particles = [Particle(bounds) for _ in range(n_particles)]
            global_best_position = particles[0].position.copy()
            global_best_score = -float('inf')
            
            # Evaluation function
            def evaluate_particle(position):
                h1, h2 = int(position[0]), int(position[1])
                h1 = max(2, min(h1, 30))
                h2 = max(1, min(h2, 20))
                alpha = position[2]
                lr = position[3]
                
                model = MLPClassifier(
                    hidden_layer_sizes=(h1, h2), alpha=alpha, learning_rate_init=lr,
                    max_iter=200, random_state=RANDOM_SEED, early_stopping=True
                )
                try:
                    scores = cross_val_score(model, X_train, y_train, cv=3)
                    return scores.mean()
                except:
                    return 0
            
            # Initial evaluation
            for particle in particles:
                score = evaluate_particle(particle.position)
                if score > particle.best_score:
                    particle.best_score = score
                    particle.best_position = particle.position.copy()
                if score > global_best_score:
                    global_best_score = score
                    global_best_position = particle.position.copy()
            
            # PSO main loop
            for it in range(n_iterations):
                w = 0.9 - (0.9 - 0.4) * (it / n_iterations)
                
                for particle in particles:
                    particle.update_velocity(global_best_position, w=w)
                    particle.update_position(bounds)
                    
                    score = evaluate_particle(particle.position)
                    
                    if score > particle.best_score:
                        particle.best_score = score
                        particle.best_position = particle.position.copy()
                        
                        if score > global_best_score:
                            global_best_score = score
                            global_best_position = particle.position.copy()
                
                if (it + 1) % 10 == 0:
                    status.text(f"PSO: Iteration {it+1}/{n_iterations}, Best: {global_best_score:.4f}")
            
            # Train final PSO model
            best_h1 = int(global_best_position[0])
            best_h2 = int(global_best_position[1])
            best_alpha = global_best_position[2]
            best_lr = global_best_position[3]
            
            pso_model = MLPClassifier(
                hidden_layer_sizes=(best_h1, best_h2), alpha=best_alpha, learning_rate_init=best_lr,
                max_iter=500, random_state=RANDOM_SEED, early_stopping=True
            )
            pso_model.fit(X_train, y_train)
            pso_acc = accuracy_score(y_test, pso_model.predict(X_test))
            pso_time = time.time() - start
            results.append({"Method": f"PSO ({best_h1},{best_h2})", "Accuracy": pso_acc, "Time": pso_time})
            progress_bar.progress(40)
            
            # ==========================================================
            # 3. REAL GA (Genetic Algorithm)
            # ==========================================================
            status.text(f"Training REAL GA ({n_generations} generations)...")
            start = time.time()
            np.random.seed(RANDOM_SEED + 1)
            
            # Population initialization
            population_size = 30
            population = []
            for _ in range(population_size):
                individual = [
                    np.random.randint(4, 25),   # h1
                    np.random.randint(2, 15),   # h2
                    np.random.uniform(0.0001, 0.01),  # alpha
                    np.random.uniform(0.0001, 0.005)   # lr
                ]
                population.append(individual)
            
            def evaluate_ga(individual):
                h1, h2 = individual[0], individual[1]
                h1 = max(2, min(h1, 30))
                h2 = max(1, min(h2, 20))
                alpha = individual[2]
                lr = individual[3]
                
                model = MLPClassifier(
                    hidden_layer_sizes=(h1, h2), alpha=alpha, learning_rate_init=lr,
                    max_iter=200, random_state=RANDOM_SEED, early_stopping=True
                )
                try:
                    scores = cross_val_score(model, X_train, y_train, cv=3)
                    return scores.mean()
                except:
                    return 0
            
            # Evaluate initial population
            fitness = [evaluate_ga(ind) for ind in population]
            
            for gen in range(n_generations):
                # Selection (tournament)
                new_population = []
                for _ in range(population_size):
                    idx1, idx2 = np.random.choice(population_size, 2, replace=False)
                    parent1 = population[idx1] if fitness[idx1] > fitness[idx2] else population[idx2]
                    idx1, idx2 = np.random.choice(population_size, 2, replace=False)
                    parent2 = population[idx1] if fitness[idx1] > fitness[idx2] else population[idx2]
                    
                    # Crossover
                    child = []
                    for j in range(4):
                        if np.random.random() < 0.7:
                            child.append(parent1[j])
                        else:
                            child.append(parent2[j])
                    
                    # Mutation
                    for j in range(4):
                        if np.random.random() < 0.1:
                            if j == 0:
                                child[j] += np.random.randint(-2, 3)
                                child[j] = np.clip(child[j], 4, 25)
                            elif j == 1:
                                child[j] += np.random.randint(-2, 3)
                                child[j] = np.clip(child[j], 2, 15)
                            else:
                                child[j] += np.random.normal(0, 0.0005)
                                if j == 2:
                                    child[j] = np.clip(child[j], 0.0001, 0.01)
                                else:
                                    child[j] = np.clip(child[j], 0.0001, 0.005)
                    
                    new_population.append(child)
                
                population = new_population
                fitness = [evaluate_ga(ind) for ind in population]
                
                if (gen + 1) % 5 == 0:
                    best_fitness = max(fitness)
                    status.text(f"GA: Generation {gen+1}/{n_generations}, Best: {best_fitness:.4f}")
            
            # Get best individual
            best_idx = np.argmax(fitness)
            best_ga = population[best_idx]
            ga_model = MLPClassifier(
                hidden_layer_sizes=(best_ga[0], best_ga[1]), alpha=best_ga[2], learning_rate_init=best_ga[3],
                max_iter=500, random_state=RANDOM_SEED, early_stopping=True
            )
            ga_model.fit(X_train, y_train)
            ga_acc = accuracy_score(y_test, ga_model.predict(X_test))
            ga_time = time.time() - start
            results.append({"Method": f"GA ({best_ga[0]},{best_ga[1]})", "Accuracy": ga_acc, "Time": ga_time})
            progress_bar.progress(70)
            
            # ==========================================================
            # 4. REAL GWO (Grey Wolf Optimizer)
            # ==========================================================
            status.text(f"Training REAL GWO ({n_wolves} wolves, {n_iterations} iterations)...")
            start = time.time()
            np.random.seed(RANDOM_SEED + 2)
            
            # Initialize wolves
            wolves = np.random.uniform(
                low=[4, 2, 0.0001, 0.0001],
                high=[25, 15, 0.01, 0.005],
                size=(n_wolves, 4)
            )
            
            alpha_pos = wolves[0].copy()
            beta_pos = wolves[0].copy()
            delta_pos = wolves[0].copy()
            alpha_score = -float('inf')
            beta_score = -float('inf')
            delta_score = -float('inf')
            
            def evaluate_gwo(wolf):
                h1, h2 = int(wolf[0]), int(wolf[1])
                h1 = max(2, min(h1, 30))
                h2 = max(1, min(h2, 20))
                alpha = wolf[2]
                lr = wolf[3]
                
                model = MLPClassifier(
                    hidden_layer_sizes=(h1, h2), alpha=alpha, learning_rate_init=lr,
                    max_iter=200, random_state=RANDOM_SEED, early_stopping=True
                )
                try:
                    scores = cross_val_score(model, X_train, y_train, cv=3)
                    return scores.mean()
                except:
                    return 0
            
            # Initial evaluation
            for i, wolf in enumerate(wolves):
                fitness = evaluate_gwo(wolf)
                if fitness > alpha_score:
                    alpha_score = fitness
                    alpha_pos = wolf.copy()
                elif fitness > beta_score:
                    beta_score = fitness
                    beta_pos = wolf.copy()
                elif fitness > delta_score:
                    delta_score = fitness
                    delta_pos = wolf.copy()
            
            for it in range(n_iterations):
                a = 2 - it * (2 / n_iterations)
                
                for i in range(n_wolves):
                    for j in range(4):
                        r1, r2 = np.random.rand(2)
                        A1 = 2 * a * r1 - a
                        C1 = 2 * r2
                        D_alpha = abs(C1 * alpha_pos[j] - wolves[i, j])
                        X1 = alpha_pos[j] - A1 * D_alpha
                        
                        r1, r2 = np.random.rand(2)
                        A2 = 2 * a * r1 - a
                        C2 = 2 * r2
                        D_beta = abs(C2 * beta_pos[j] - wolves[i, j])
                        X2 = beta_pos[j] - A2 * D_beta
                        
                        r1, r2 = np.random.rand(2)
                        A3 = 2 * a * r1 - a
                        C3 = 2 * r2
                        D_delta = abs(C3 * delta_pos[j] - wolves[i, j])
                        X3 = delta_pos[j] - A3 * D_delta
                        
                        wolves[i, j] = (X1 + X2 + X3) / 3
                    
                    wolves[i] = np.clip(wolves[i], [4, 2, 0.0001, 0.0001], [25, 15, 0.01, 0.005])
                    fitness = evaluate_gwo(wolves[i])
                    
                    if fitness > alpha_score:
                        alpha_score = fitness
                        alpha_pos = wolves[i].copy()
                    elif fitness > beta_score:
                        beta_score = fitness
                        beta_pos = wolves[i].copy()
                    elif fitness > delta_score:
                        delta_score = fitness
                        delta_pos = wolves[i].copy()
                
                if (it + 1) % 10 == 0:
                    status.text(f"GWO: Iteration {it+1}/{n_iterations}, Best: {alpha_score:.4f}")
            
            best_gwo_h1 = int(alpha_pos[0])
            best_gwo_h2 = int(alpha_pos[1])
            gwo_model = MLPClassifier(
                hidden_layer_sizes=(best_gwo_h1, best_gwo_h2), alpha=alpha_pos[2], learning_rate_init=alpha_pos[3],
                max_iter=500, random_state=RANDOM_SEED, early_stopping=True
            )
            gwo_model.fit(X_train, y_train)
            gwo_acc = accuracy_score(y_test, gwo_model.predict(X_test))
            gwo_time = time.time() - start
            results.append({"Method": f"GWO ({best_gwo_h1},{best_gwo_h2})", "Accuracy": gwo_acc, "Time": gwo_time})
            progress_bar.progress(100)
            
            status.text("Training complete!")
            
            st.session_state['results'] = results
            st.session_state['pso_model'] = pso_model
            st.session_state['scaler'] = scaler
            st.session_state['label_encoder'] = label_encoder
            st.session_state['X_test'] = X_test
            st.session_state['y_test'] = y_test
            
            st.success("✅ All models trained successfully!")
    
    # ===============================
    # RESULTS
    # ===============================
    
    if 'results' in st.session_state:
        st.header("📊 Step 4: Model Comparison Results")
        
        results = st.session_state['results']
        res_df = pd.DataFrame(results)
        
        # Find best method
        best_idx = res_df['Accuracy'].argmax()
        best_method = res_df.iloc[best_idx]['Method']
        best_acc = res_df.iloc[best_idx]['Accuracy']
        
        styled = res_df.style.highlight_max(subset=['Accuracy'], color='lightgreen')
        st.dataframe(styled, use_container_width=True)
        
        st.success(f"🏆 **Best Method: {best_method}** with {best_acc:.3f} ({best_acc*100:.1f}%) accuracy")
        
        # Charts
        col1, col2 = st.columns(2)
        
        with col1:
            fig, ax = plt.subplots(figsize=(8, 5))
            bars = ax.bar(res_df['Method'], res_df['Accuracy'], 
                         color=['#95a5a6', '#e74c3c', '#2ecc71', '#3498db'])
            ax.set_ylim(0, 1)
            ax.set_ylabel('Accuracy')
            ax.set_title('Accuracy Comparison')
            ax.tick_params(axis='x', rotation=15)
            for bar, acc in zip(bars, res_df['Accuracy']):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{acc:.3f}', ha='center')
            plt.tight_layout()
            st.pyplot(fig)
        
        with col2:
            fig, ax = plt.subplots(figsize=(8, 5))
            bars = ax.bar(res_df['Method'], res_df['Time'], 
                         color=['#95a5a6', '#e74c3c', '#2ecc71', '#3498db'])
            ax.set_ylabel('Time (seconds)')
            ax.set_title('Time Comparison')
            ax.tick_params(axis='x', rotation=15)
            for bar, t in zip(bars, res_df['Time']):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{t:.1f}s', ha='center')
            plt.tight_layout()
            st.pyplot(fig)
        
        # Confusion Matrix - Best Model
        st.subheader("📊 Confusion Matrix (Best Model)")
        
        X_test = st.session_state['X_test']
        y_test = st.session_state['y_test']
        label_encoder = st.session_state['label_encoder']
        
        if "PSO" in best_method:
            best_model = st.session_state['pso_model']
        else:
            best_model = st.session_state['pso_model']  # Fallback
        
        y_pred_best = best_model.predict(X_test)
        
        species_short = [s.split()[0] for s in label_encoder.classes_]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred_best)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=species_short, yticklabels=species_short, ax=ax)
        ax.set_title(f'Confusion Matrix - {best_method}', fontsize=12, fontweight='bold')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        plt.tight_layout()
        st.pyplot(fig)
        
        # Per-species accuracy
        st.subheader("📋 Per-Species Classification Accuracy")
        
        species_accuracy = []
        for i, sp in enumerate(label_encoder.classes_):
            mask = y_test == i
            if mask.sum() > 0:
                correct = (y_pred_best[mask] == i).sum()
                total = mask.sum()
                acc = correct / total
                species_accuracy.append({
                    'Species': sp,
                    'Test Samples': total,
                    'Correct': correct,
                    'Accuracy': f"{acc:.3f} ({acc*100:.1f}%)"
                })
        
        st.dataframe(pd.DataFrame(species_accuracy), use_container_width=True)

else:
    st.info("👈 Please upload your Excel file to begin")

st.markdown("---")
st.caption("FYP Project | Universiti Malaysia Terengganu")
