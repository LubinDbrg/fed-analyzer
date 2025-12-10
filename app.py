import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import io
import re

st.set_page_config(page_title="KPI Finance - Hybride AI", layout="wide")

# --- 1. FONCTIONS DE LECTURE (INCHANGÉES) ---
def detect_separator(line):
    if line.count(';') > line.count(','): return ';'
    if line.count('|') > line.count(';'): return '|'
    return ','

def standardize_columns(df):
    mapping = {
        'EcritureDate': ['EcritureDate', 'DateEcriture', 'Date', 'date_ecriture'],
        'CompteNum': ['CompteNum', 'NumCompte', 'Compte', 'NumeroCompte'],
        'Debit': ['Debit', 'MontantDebit', 'MntDebit', 'Débit'],
        'Credit': ['Credit', 'MontantCredit', 'MntCredit', 'Crédit']
    }
    clean_cols = {c: c.strip() for c in df.columns}
    df = df.rename(columns=clean_cols)
    final_rename = {}
    for col in df.columns:
        for standard, variants in mapping.items():
            if any(v.lower() == col.lower() for v in variants):
                final_rename[col] = standard
                break
    if final_rename: df = df.rename(columns=final_rename)
    return df

def clean_financial_number(series):
    s = series.astype(str).str.replace(r'\s+', '', regex=True)
    s = s.str.replace(',', '.', regex=False)
    return pd.to_numeric(s, errors='coerce').fillna(0.0)

def load_fec_robust(uploaded_file):
    try:
        bytes_data = uploaded_file.getvalue()
        try: content = bytes_data.decode('latin-1')
        except: content = bytes_data.decode('utf-8', errors='ignore')
        
        first_line = content.split('\n')[0]
        sep = detect_separator(first_line)
        
        df = pd.read_csv(io.StringIO(content), sep=sep, dtype=str)
        df = standardize_columns(df)
        
        required = ['CompteNum', 'Debit', 'Credit']
        if not all(col in df.columns for col in required): return None

        df['MontantDebit'] = clean_financial_number(df['Debit'])
        df['MontantCredit'] = clean_financial_number(df['Credit'])
        df['CompteNum'] = df['CompteNum'].astype(str).str.replace(r'\D', '', regex=True)

        if 'EcritureDate' in df.columns:
            df['Date_Analyse'] = pd.to_datetime(df['EcritureDate'], format='%Y%m%d', errors='coerce')
            mask_nat = df['Date_Analyse'].isna()
            if mask_nat.any():
                df.loc[mask_nat, 'Date_Analyse'] = pd.to_datetime(df.loc[mask_nat, 'EcritureDate'], dayfirst=True, errors='coerce')
            df = df.dropna(subset=['Date_Analyse'])
        else: return None

        return df
    except: return None

# --- 2. CALCUL DES INDICATEURS ---
def calculer_indicateurs_mensuels(df):
    if df.empty: return pd.DataFrame()
    df = df.set_index('Date_Analyse').sort_index()
    groupe_mois = df.groupby(pd.Grouper(freq='ME'))
    
    resultats = []
    for mois, data in groupe_mois:
        if data.empty:
            resultats.append({'Date': mois, 'CA': 0, 'EBITDA': 0, 'Resultat': 0})
            continue

        mask_ca = data['CompteNum'].str.startswith('70')
        ca = (data.loc[mask_ca, 'MontantCredit'] - data.loc[mask_ca, 'MontantDebit']).sum()

        mask_prod = data['CompteNum'].str.match(r'^(70|71|72|73|74)')
        prod = (data.loc[mask_prod, 'MontantCredit'] - data.loc[mask_prod, 'MontantDebit']).sum()
        mask_chg = data['CompteNum'].str.match(r'^(60|61|62|63|64)')
        chg = (data.loc[mask_chg, 'MontantDebit'] - data.loc[mask_chg, 'MontantCredit']).sum()
        ebitda = prod - chg

        mask_cl7 = data['CompteNum'].str.startswith('7')
        total_prod = (data.loc[mask_cl7, 'MontantCredit'] - data.loc[mask_cl7, 'MontantDebit']).sum()
        mask_cl6 = data['CompteNum'].str.startswith('6')
        total_chg = (data.loc[mask_cl6, 'MontantDebit'] - data.loc[mask_cl6, 'MontantCredit']).sum()
        resultat = total_prod - total_chg

        resultats.append({'Date': mois, 'CA': ca, 'EBITDA': ebitda, 'Resultat': resultat})

    return pd.DataFrame(resultats).set_index('Date')

def calculer_tresorerie_quotidienne(df):
    if df.empty: return pd.Series()
    mask_treso = df['CompteNum'].str.startswith('5')
    df_treso = df[mask_treso].copy()
    df_treso['Flux'] = df_treso['MontantDebit'] - df_treso['MontantCredit']
    df_treso = df_treso.set_index('Date_Analyse').sort_index()
    flux_journalier = df_treso['Flux'].resample('D').sum().fillna(0)
    return flux_journalier.cumsum()

# --- 3. PRÉDICTION HYBRIDE (LINEAR + RANDOM FOREST) ---

def create_features(df, label=None):
    """
    Crée des caractéristiques temporelles pour le Random Forest
    """
    df = df.copy()
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    df['dayofyear'] = df.index.dayofyear
    # On ajoute un index numérique simple pour la régression linéaire
    df['time_idx'] = np.arange(len(df))
    return df

def predict_hybrid(series, months_to_predict, trend_factor=1.0):
    """
    Modèle Hybride :
    1. LinearRegression capture la TENDANCE globale (hausse/baisse long terme).
    2. RandomForest capture la SAISONNALITÉ (les cycles que la ligne droite rate).
    """
    if len(series) < 6: return None

    # Préparation des données
    df = pd.DataFrame({'y': series})
    df = create_features(df)
    
    # Séparation Features (X) / Target (y)
    X = df[['time_idx', 'month', 'quarter']] # Features simples
    y = df['y']

    # --- ÉTAPE 1 : Apprendre la Tendance (Linear Regression) ---
    # On utilise seulement time_idx pour la tendance pure
    model_trend = LinearRegression()
    model_trend.fit(df[['time_idx']], y)
    trend_pred = model_trend.predict(df[['time_idx']])
    
    # --- ÉTAPE 2 : Apprendre les Résidus (Ce que la tendance rate) ---
    # Résidu = Réalité - Tendance
    y_residuals = y - trend_pred
    
    # Le Random Forest va apprendre à prédire ces écarts (saisonnalité)
    model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
    model_rf.fit(X, y_residuals) # Il apprend sur tout X (mois, trimestre...)

    # --- ÉTAPE 3 : Prédiction Future ---
    last_date = series.index[-1]
    future_dates = pd.date_range(start=last_date, periods=months_to_predict + 1, freq=series.index.freq)[1:]
    
    future_df = pd.DataFrame(index=future_dates)
    # On recrée les mêmes features pour le futur
    future_df['month'] = future_df.index.month
    future_df['quarter'] = future_df.index.quarter
    future_df['dayofyear'] = future_df.index.dayofyear
    # time_idx continue après la fin de l'historique
    last_idx = df['time_idx'].iloc[-1]
    future_df['time_idx'] = np.arange(last_idx + 1, last_idx + 1 + months_to_predict)
    
    X_future = future_df[['time_idx', 'month', 'quarter']]
    
    # A. Prédire la tendance future
    future_trend = model_trend.predict(future_df[['time_idx']])
    
    # B. Prédire la saisonnalité future (correction)
    future_residuals = model_rf.predict(X_future)
    
    # C. Combiner les deux
    final_pred = future_trend + future_residuals
    
    # D. Appliquer le scénario économique (facteur externe)
    growth_curve = np.linspace(1, trend_factor, len(final_pred))
    final_pred = final_pred * growth_curve
    
    return pd.Series(final_pred, index=future_dates)

# --- 4. INTERFACE ---

st.sidebar.header("Paramètres")
api_key = st.sidebar.text_input("Clé API Gemini", type="password")
uploaded_files = st.sidebar.file_uploader("Fichiers FEC", accept_multiple_files=True)
horizon_years = st.sidebar.slider("Horizon prédiction (années)", 1, 3, 2)

st.sidebar.subheader("🌍 Scénario")
scenario_map = {
    "Neutre": 1.0, "Optimiste (+5%)": 1.05, "Pessimiste (-5%)": 0.95,
    "Inflation (+1.5%)": 1.015
}
choix = st.sidebar.selectbox("Tendance :", list(scenario_map.keys()))
trend_factor = scenario_map[choix]

st.title("📊 Finance : Modèle Hybride (AI)")

if uploaded_files:
    all_dfs = []
    for file in uploaded_files:
        df = load_fec_robust(file)
        if df is not None: all_dfs.append(df)

    if all_dfs:
        df_global = pd.concat(all_dfs, ignore_index=True)
        df_mensuel = calculer_indicateurs_mensuels(df_global)
        serie_treso_jour = calculer_tresorerie_quotidienne(df_global)

        if not df_mensuel.empty:
            months_pred = horizon_years * 12

            last_m = df_mensuel.iloc[-1]
            last_treso = serie_treso_jour.iloc[-1] if not serie_treso_jour.empty else 0
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("CA Mensuel", f"{last_m['CA']:,.0f} €")
            c2.metric("EBITDA", f"{last_m['EBITDA']:,.0f} €")
            c3.metric("Résultat Net", f"{last_m['Resultat']:,.0f} €")
            c4.metric("Trésorerie J-J", f"{last_treso:,.0f} €")
            
            st.markdown("---")

            col1, col2 = st.columns(2)

            # GRAPHIQUE 1 : CA (Barres)
            with col1:
                fig_ca = go.Figure()
                fig_ca.add_trace(go.Bar(x=df_mensuel.index, y=df_mensuel['CA'], name='Historique', marker_color='#1f77b4'))
                
                # Prédiction Hybride
                pred_ca = predict_hybrid(df_mensuel['CA'], months_pred, trend_factor)
                if pred_ca is not None:
                    fig_ca.add_trace(go.Bar(x=pred_ca.index, y=pred_ca, name='Prévision AI', marker_pattern_shape='/', marker_color='#1f77b4', opacity=0.5))
                
                fig_ca.update_layout(title="Chiffre d'Affaires (Hybride Linear+RF)", height=350)
                st.plotly_chart(fig_ca, use_container_width=True)

            # GRAPHIQUE 2 : EBITDA (Courbe)
            with col2:
                fig_eb = go.Figure()
                fig_eb.add_trace(go.Scatter(x=df_mensuel.index, y=df_mensuel['EBITDA'], mode='lines', name='Historique', 
                                            line=dict(color='#ff7f0e', width=3, shape='spline', smoothing=1.3)))
                
                pred_eb = predict_hybrid(df_mensuel['EBITDA'], months_pred, trend_factor)
                if pred_eb is not None:
                     fig_eb.add_trace(go.Scatter(x=pred_eb.index, y=pred_eb, mode='lines', name='Prévision AI', 
                                                 line=dict(color='#ff7f0e', width=3, dash='dot', shape='spline')))
                
                fig_eb.update_layout(title="EBITDA", height=350)
                st.plotly_chart(fig_eb, use_container_width=True)

            col3, col4 = st.columns(2)

            # GRAPHIQUE 3 : RESULTAT
            with col3:
                fig_res = go.Figure()
                colors = ['#2ca02c' if v >= 0 else '#d62728' for v in df_mensuel['Resultat']]
                fig_res.add_trace(go.Bar(x=df_mensuel.index, y=df_mensuel['Resultat'], name='Historique', marker_color=colors))
                
                pred_res = predict_hybrid(df_mensuel['Resultat'], months_pred, trend_factor)
                if pred_res is not None:
                    fig_res.add_trace(go.Bar(x=pred_res.index, y=pred_res, name='Prévision AI', marker_pattern_shape='/', marker_color='#7f7f7f', opacity=0.5))

                fig_res.update_layout(title="Résultat Net", height=350)
                st.plotly_chart(fig_res, use_container_width=True)

            # GRAPHIQUE 4 : TRÉSORERIE (On garde Holt-Winters ou Linear sur la haute fréquence pour la rapidité)
            # Pour la démo, on utilise le resampling mensuel pour l'AI hybride
            with col4:
                fig_tr = go.Figure()
                fig_tr.add_trace(go.Scatter(x=serie_treso_jour.index, y=serie_treso_jour, mode='lines', name='Historique', fill='tozeroy', line=dict(color='#9467bd', width=1)))

                # On prédit sur la base mensuelle pour éviter d'avoir un modèle trop lourd
                treso_mensuelle = serie_treso_jour.resample('ME').last()
                pred_tr = predict_hybrid(treso_mensuelle, months_pred, trend_factor)
                
                if pred_tr is not None:
                     fig_tr.add_trace(go.Scatter(x=pred_tr.index, y=pred_tr, mode='lines', name='Prévision AI', 
                                                 line=dict(color='#9467bd', width=2, dash='dot')))

                fig_tr.update_layout(title="Trésorerie", height=350)
                st.plotly_chart(fig_tr, use_container_width=True)
