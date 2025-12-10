import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import io
import re

st.set_page_config(page_title="KPI Finance - Hybride AI", layout="wide")

# --- 1. FONCTIONS DE LECTURE ---
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
    # Utilisation de 'ME' (Month End) pour grouper
    groupe_mois = df.groupby(pd.Grouper(freq='ME'))
    
    resultats = []
    for mois, data in groupe_mois:
        if data.empty:
            # On garde les mois vides pour la continuité temporelle (important pour les graphes)
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

# --- 3. PRÉDICTION HYBRIDE (ROBUSTE) ---

def create_features(df, label=None):
    df = df.copy()
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    df['dayofyear'] = df.index.dayofyear
    df['time_idx'] = np.arange(len(df))
    return df

def predict_hybrid_ca(series, months_to_predict, trend_factor=1.0):
    """
    Prédit le CA (Top Line) en utilisant AI + Scénario.
    Version 'Tolérante' : Accepte moins de données et gère les erreurs.
    """
    # Nettoyage préventif
    series = series.fillna(0)
    
    # 1. Vérification minimale : On essaie de prédire même avec peu de points (min 2 mois)
    if len(series) < 2:
        return None

    try:
        # Préparation des données
        df = pd.DataFrame({'y': series})
        df = create_features(df)
        
        X = df[['time_idx', 'month', 'quarter']]
        y = df['y']

        # 2. Modèle de Tendance (Linear)
        model_trend = LinearRegression()
        model_trend.fit(df[['time_idx']], y)
        trend_pred = model_trend.predict(df[['time_idx']])
        
        # 3. Modèle de Saisonnalité (Random Forest sur les résidus)
        y_residuals = y - trend_pred
        model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
        model_rf.fit(X, y_residuals)

        # 4. Génération du Futur
        last_date = series.index[-1]
        # Force la fréquence 'ME' pour éviter les erreurs si l'index est malformé
        future_dates = pd.date_range(start=last_date, periods=months_to_predict + 1, freq='ME')[1:]
        
        future_df = pd.DataFrame(index=future_dates)
        future_df['month'] = future_df.index.month
        future_df['quarter'] = future_df.index.quarter
        future_df['dayofyear'] = future_df.index.dayofyear
        
        last_idx = df['time_idx'].iloc[-1]
        future_df['time_idx'] = np.arange(last_idx + 1, last_idx + 1 + months_to_predict)
        
        X_future = future_df[['time_idx', 'month', 'quarter']]
        
        future_trend = model_trend.predict(future_df[['time_idx']])
        future_residuals = model_rf.predict(X_future)
        
        final_pred = future_trend + future_residuals
        
        # Application du scénario
        growth_curve = np.linspace(1, trend_factor, len(final_pred))
        final_pred = final_pred * growth_curve
        
        # On évite les prédictions négatives pour le CA
        final_pred = np.maximum(final_pred, 0)
        
        return pd.Series(final_pred, index=future_dates)
        
    except Exception as e:
        # En cas d'erreur interne, on l'affiche dans la console Streamlit pour débogage
        print(f"Erreur prédiction: {e}")
        return None

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

st.title("📊 Finance : Modèle Hybride Cohérent")

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

            # --- CALCUL DES PRÉDICTIONS ---
            with st.spinner('Calcul des prédictions IA en cours...'):
                pred_ca = predict_hybrid_ca(df_mensuel['CA'], months_pred, trend_factor)
            
            # Variables pour stocker les prédictions dérivées
            pred_ebitda = None
            pred_result = None
            pred_treso = None

            if pred_ca is not None:
                # Calcul des dérivés seulement si la prédiction CA a réussi
                last_12 = df_mensuel.iloc[-12:] if len(df_mensuel) >= 12 else df_mensuel
                
                # Marge EBITDA moyenne
                sum_ca = last_12['CA'].sum()
                marge_ebitda = (last_12['EBITDA'].sum() / sum_ca) if sum_ca != 0 else 0
                pred_ebitda = pred_ca * marge_ebitda

                # Écart Résultat moyen
                ecart_resultat = (last_12['EBITDA'] - last_12['Resultat']).mean()
                pred_result = pred_ebitda - ecart_resultat

                # Trésorerie Cumulative
                pred_treso_list = []
                current_cash = last_treso
                for res in pred_result:
                    current_cash += res
                    pred_treso_list.append(current_cash)
                pred_treso = pd.Series(pred_treso_list, index=pred_ca.index)
            else:
                st.warning("⚠️ Impossible de générer une prédiction : Historique insuffisant ou données trop irrégulières.")

            # --- GRAPHIQUES ---
            col1, col2 = st.columns(2)

            # GRAPHIQUE 1 : CA
            with col1:
                fig_ca = go.Figure()
                fig_ca.add_trace(go.Bar(x=df_mensuel.index, y=df_mensuel['CA'], name='Historique', marker_color='#1f77b4'))
                if pred_ca is not None:
                    fig_ca.add_trace(go.Bar(x=pred_ca.index, y=pred_ca, name='Prévision AI', marker_pattern_shape='/', marker_color='#1f77b4', opacity=0.5))
                fig_ca.update_layout(title="Chiffre d'Affaires", height=350, template="plotly_dark", margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_ca, use_container_width=True)

            # GRAPHIQUE 2 : EBITDA
            with col2:
                fig_eb = go.Figure()
                fig_eb.add_trace(go.Scatter(x=df_mensuel.index, y=df_mensuel['EBITDA'], mode='lines+markers', name='Historique', 
                                            line=dict(color='#ff7f0e', width=3)))
                if pred_ebitda is not None:
                     fig_eb.add_trace(go.Scatter(x=pred_ebitda.index, y=pred_ebitda, mode='lines+markers', name='Prévision', 
                                                 line=dict(color='#ff7f0e', width=2, dash='dot')))
                fig_eb.add_hline(y=0, line_color="white", line_width=1, opacity=0.3)
                fig_eb.update_layout(title="EBITDA", height=350, template="plotly_dark", margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_eb, use_container_width=True)

            col3, col4 = st.columns(2)

            # GRAPHIQUE 3 : RESULTAT
            with col3:
                fig_res = go.Figure()
                colors_hist = ['#2ca02c' if v >= 0 else '#d62728' for v in df_mensuel['Resultat']]
                fig_res.add_trace(go.Bar(x=df_mensuel.index, y=df_mensuel['Resultat'], name='Historique', marker_color=colors_hist))
                
                if pred_result is not None:
                    colors_pred = ['#2ca02c' if v >= 0 else '#d62728' for v in pred_result]
                    fig_res.add_trace(go.Bar(x=pred_result.index, y=pred_result, name='Prévision', marker_pattern_shape='/', marker_color=colors_pred, opacity=0.6))

                fig_res.add_hline(y=0, line_color="white", line_width=1, opacity=0.3)
                fig_res.update_layout(title="Résultat Net", height=350, template="plotly_dark", margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_res, use_container_width=True)

            # GRAPHIQUE 4 : TRÉSORERIE (CORRIGÉ LISSAGE)
            with col4:
                fig_tr = go.Figure()
                
                # CORRECTION ESCALIER : On rééchantillonne l'historique au MOIS pour lisser la courbe
                # Cela relie les points de fin de mois entre eux, supprimant les "marches" quotidiennes
                treso_lisse = serie_treso_jour.resample('ME').last()
                
                fig_tr.add_trace(go.Scatter(x=treso_lisse.index, y=treso_lisse, mode='lines', name='Historique', 
                                          fill='tozeroy', line=dict(color='#9467bd', width=2)))

                if pred_treso is not None:
                     fig_tr.add_trace(go.Scatter(x=pred_treso.index, y=pred_treso, mode='lines', name='Prévision', 
                                                 fill='tozeroy', line=dict(color='#9467bd', width=2, dash='dot')))
                
                fig_tr.add_hline(y=0, line_color="red", line_width=1, line_dash="dot")
                fig_tr.update_layout(title="Trésorerie (Lissée & Projetée)", height=350, template="plotly_dark", margin=dict(l=20, r=20, t=40, b=20))
                st.plotly_chart(fig_tr, use_container_width=True)
