import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.linear_model import LinearRegression
import io
import re

st.set_page_config(page_title="KPI Finance - Courbes Lisses", layout="wide")

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

# --- 2. CALCUL DES INDICATEURS (AMÉLIORÉ) ---

def calculer_indicateurs_mensuels(df):
    """Calcule CA, EBITDA, RESULTAT au mois (standard comptable)"""
    if df.empty: return pd.DataFrame()
    df = df.set_index('Date_Analyse').sort_index()
    groupe_mois = df.groupby(pd.Grouper(freq='ME'))
    
    resultats = []
    for mois, data in groupe_mois:
        if data.empty:
            resultats.append({'Date': mois, 'CA': 0, 'EBITDA': 0, 'Resultat': 0})
            continue

        # CA
        mask_ca = data['CompteNum'].str.startswith('70')
        ca = (data.loc[mask_ca, 'MontantCredit'] - data.loc[mask_ca, 'MontantDebit']).sum()

        # EBITDA
        mask_prod = data['CompteNum'].str.match(r'^(70|71|72|73|74)')
        prod = (data.loc[mask_prod, 'MontantCredit'] - data.loc[mask_prod, 'MontantDebit']).sum()
        mask_chg = data['CompteNum'].str.match(r'^(60|61|62|63|64)')
        chg = (data.loc[mask_chg, 'MontantDebit'] - data.loc[mask_chg, 'MontantCredit']).sum()
        ebitda = prod - chg

        # Résultat
        mask_cl7 = data['CompteNum'].str.startswith('7')
        total_prod = (data.loc[mask_cl7, 'MontantCredit'] - data.loc[mask_cl7, 'MontantDebit']).sum()
        mask_cl6 = data['CompteNum'].str.startswith('6')
        total_chg = (data.loc[mask_cl6, 'MontantDebit'] - data.loc[mask_cl6, 'MontantCredit']).sum()
        resultat = total_prod - total_chg

        resultats.append({'Date': mois, 'CA': ca, 'EBITDA': ebitda, 'Resultat': resultat})

    return pd.DataFrame(resultats).set_index('Date')

def calculer_tresorerie_quotidienne(df):
    """
    NOUVEAU : Calcule la trésorerie jour par jour pour éviter l'effet escalier.
    C'est beaucoup plus précis que le mensuel.
    """
    if df.empty: return pd.Series()
    
    # On ne garde que les comptes de classe 5 (Banque/Caisse)
    mask_treso = df['CompteNum'].str.startswith('5')
    df_treso = df[mask_treso].copy()
    
    # Calcul du flux net pour chaque ligne
    df_treso['Flux'] = df_treso['MontantDebit'] - df_treso['MontantCredit']
    
    # On indexe par date
    df_treso = df_treso.set_index('Date_Analyse').sort_index()
    
    # 1. On regroupe par JOUR ('D') et on fait la somme des flux du jour
    flux_journalier = df_treso['Flux'].resample('D').sum().fillna(0)
    
    # 2. On calcule le cumul (Solde en banque)
    solde_journalier = flux_journalier.cumsum()
    
    return solde_journalier

# --- 3. PRÉDICTIONS ---

def generate_predictions_advanced(series, months_to_predict, trend_factor=1.0):
    if len(series) < 6: return None

    # On prépare l'index futur
    last_date = series.index[-1]
    future_dates = pd.date_range(start=last_date, periods=months_to_predict + 1, freq=series.index.freq)[1:]
    
    # Option A: Holt-Winters (Saisonnalité)
    pred_values = None
    if len(series) >= 20:
        try:
            # On détecte la fréquence pour ajuster la période saisonnière
            seasonal_periods = 12 if series.index.freqstr == 'ME' else 30 # Approx 30 jours si journalier
            model = ExponentialSmoothing(series, trend='add', seasonal='add', seasonal_periods=seasonal_periods).fit()
            pred_values = model.forecast(months_to_predict)
        except: pass

    # Option B: Régression Linéaire
    if pred_values is None:
        X = np.arange(len(series)).reshape(-1, 1)
        y = series.values
        reg = LinearRegression().fit(X, y)
        future_X = np.arange(len(series), len(series) + months_to_predict).reshape(-1, 1)
        pred_values = reg.predict(future_X)
        noise = np.random.normal(0, series.std() * 0.05, size=len(pred_values))
        pred_values = pred_values + noise

    # Facteur de tendance
    growth = np.linspace(1, trend_factor, len(pred_values))
    final_pred = pd.Series(pred_values * growth, index=future_dates)
    
    return final_pred

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

st.title("📊 Finance : Haute Résolution")

if uploaded_files:
    all_dfs = []
    for file in uploaded_files:
        df = load_fec_robust(file)
        if df is not None: all_dfs.append(df)

    if all_dfs:
        df_global = pd.concat(all_dfs, ignore_index=True)
        
        # 1. Calcul Mensuel pour P&L (CA, EBITDA, Résultat)
        df_mensuel = calculer_indicateurs_mensuels(df_global)
        
        # 2. Calcul QUOTIDIEN pour Trésorerie (Beaucoup plus de points !)
        serie_treso_jour = calculer_tresorerie_quotidienne(df_global)

        if not df_mensuel.empty:
            months_pred = horizon_years * 12
            days_pred = horizon_years * 365 # Pour la tréso jour

            # Cards
            last_m = df_mensuel.iloc[-1]
            last_treso = serie_treso_jour.iloc[-1] if not serie_treso_jour.empty else 0
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("CA Mensuel", f"{last_m['CA']:,.0f} €")
            c2.metric("EBITDA", f"{last_m['EBITDA']:,.0f} €")
            c3.metric("Résultat Net", f"{last_m['Resultat']:,.0f} €")
            c4.metric("Trésorerie J-J", f"{last_treso:,.0f} €")
            
            st.markdown("---")

            col1, col2 = st.columns(2)

            # --- GRAPHIQUE 1 : CA (Barres) ---
            with col1:
                fig_ca = go.Figure()
                fig_ca.add_trace(go.Bar(x=df_mensuel.index, y=df_mensuel['CA'], name='Historique', marker_color='#1f77b4'))
                
                # Prédiction CA
                pred_ca = generate_predictions_advanced(df_mensuel['CA'], months_pred, trend_factor)
                if pred_ca is not None:
                    fig_ca.add_trace(go.Bar(x=pred_ca.index, y=pred_ca, name='Prévision', marker_pattern_shape='/', marker_color='#1f77b4', opacity=0.5))
                
                fig_ca.update_layout(title="Chiffre d'Affaires (Barres)", height=350)
                st.plotly_chart(fig_ca, use_container_width=True)

            # --- GRAPHIQUE 2 : EBITDA (Courbe LISSÉE) ---
            with col2:
                fig_eb = go.Figure()
                # ASTUCE : line_shape='spline' pour arrondir les angles !
                fig_eb.add_trace(go.Scatter(x=df_mensuel.index, y=df_mensuel['EBITDA'], mode='lines', name='Historique', 
                                            line=dict(color='#ff7f0e', width=3, shape='spline', smoothing=1.3)))
                
                pred_eb = generate_predictions_advanced(df_mensuel['EBITDA'], months_pred, trend_factor)
                if pred_eb is not None:
                     fig_eb.add_trace(go.Scatter(x=pred_eb.index, y=pred_eb, mode='lines', name='Prévision', 
                                                 line=dict(color='#ff7f0e', width=3, dash='dot', shape='spline')))
                
                fig_eb.update_layout(title="EBITDA (Courbe Lissée)", height=350)
                st.plotly_chart(fig_eb, use_container_width=True)

            col3, col4 = st.columns(2)

            # --- GRAPHIQUE 3 : RESULTAT (Barres couleurs) ---
            with col3:
                fig_res = go.Figure()
                colors = ['#2ca02c' if v >= 0 else '#d62728' for v in df_mensuel['Resultat']]
                fig_res.add_trace(go.Bar(x=df_mensuel.index, y=df_mensuel['Resultat'], name='Historique', marker_color=colors))
                
                pred_res = generate_predictions_advanced(df_mensuel['Resultat'], months_pred, trend_factor)
                if pred_res is not None:
                    fig_res.add_trace(go.Bar(x=pred_res.index, y=pred_res, name='Prévision', marker_pattern_shape='/', marker_color='#7f7f7f', opacity=0.5))

                fig_res.update_layout(title="Résultat Net", height=350)
                st.plotly_chart(fig_res, use_container_width=True)

            # --- GRAPHIQUE 4 : TRÉSORERIE (QUOTIDIENNE) ---
            with col4:
                fig_tr = go.Figure()
                # On trace la série quotidienne : beaucoup de points = courbe très précise
                # On peut réduire le nombre de points affichés si trop lourd avec .resample('W') si besoin, mais 'D' est top.
                fig_tr.add_trace(go.Scatter(
                    x=serie_treso_jour.index, 
                    y=serie_treso_jour, 
                    mode='lines', 
                    name='Historique J-J',
                    fill='tozeroy', # Effet d'aire remplie
                    line=dict(color='#9467bd', width=1) # Ligne fine car beaucoup de points
                ))

                # Prédiction (simplifiée par resampling pour aller plus vite)
                # On prédit sur une base mensuelle et on interpole ou on prédit jour par jour (lourd)
                # Pour l'UI, on prédit mensuellement et on affiche
                pred_tr = generate_predictions_advanced(serie_treso_jour.resample('ME').last(), months_pred, trend_factor)
                
                if pred_tr is not None:
                     fig_tr.add_trace(go.Scatter(x=pred_tr.index, y=pred_tr, mode='lines', name='Prévision (Tendance)', 
                                                 line=dict(color='#9467bd', width=2, dash='dot')))

                fig_tr.update_layout(title="Trésorerie (Haute Précision Quotidienne)", height=350)
                st.plotly_chart(fig_tr, use_container_width=True)
