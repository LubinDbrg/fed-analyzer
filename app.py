import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import google.generativeai as genai
import io
import re

# --- CONFIGURATION PAGE ---
st.set_page_config(page_title="KPI Finance & Prédictions", layout="wide")

# --- 1. FONCTIONS DE NETTOYAGE ROBUSTES ---

def detect_separator(line):
    if line.count(';') > line.count(','): return ';'
    if line.count('|') > line.count(';'): return '|'
    if line.count('\t') > 0: return '\t'
    return ','

def standardize_columns(df):
    """
    Renomme les colonnes du fichier utilisateur vers un standard interne.
    """
    # Dictionnaire des synonymes possibles
    mapping = {
        # Standard interne : [Liste des variantes possibles dans le CSV]
        'EcritureDate': ['EcritureDate', 'DateEcriture', 'Date', 'EcritureDate', 'date_ecriture'],
        'CompteNum': ['CompteNum', 'NumCompte', 'Compte', 'GeneralAccount', 'NumeroCompte'],
        'Debit': ['Debit', 'MontantDebit', 'MntDebit', 'Débit', 'DebitAmount'],
        'Credit': ['Credit', 'MontantCredit', 'MntCredit', 'Crédit', 'CreditAmount']
    }
    
    # Nettoyage des noms de colonnes du CSV (enlever espaces et mettre en minuscule pour comparer)
    clean_cols = {c: c.strip() for c in df.columns}
    df = df.rename(columns=clean_cols)
    
    final_rename = {}
    for col in df.columns:
        for standard, variants in mapping.items():
            # On cherche si la colonne actuelle ressemble à une des variantes (insensible à la casse)
            if any(v.lower() == col.lower() for v in variants):
                final_rename[col] = standard
                break
    
    if final_rename:
        df = df.rename(columns=final_rename)
        
    return df

def clean_financial_number(series):
    """
    Convertit une colonne de texte (ex: "1 500,50") en float (1500.50).
    Gère les espaces insécables et les virgules.
    """
    # Convertir en string, enlever les espaces (milliers), remplacer virgule par point
    s = series.astype(str).str.replace(r'\s+', '', regex=True) # Enlève tous les espaces
    s = s.str.replace(',', '.', regex=False)
    # Convertir en nombre, les erreurs deviennent 0
    return pd.to_numeric(s, errors='coerce').fillna(0.0)

def load_fec_robust(uploaded_file):
    try:
        # 1. Lecture brute pour détecter le format
        bytes_data = uploaded_file.getvalue()
        
        # Tentative de décodage (Latin-1 est courant pour FEC, sinon UTF-8)
        try:
            content = bytes_data.decode('latin-1')
        except:
            content = bytes_data.decode('utf-8', errors='ignore')
            
        # Détection séparateur sur la première ligne
        first_line = content.split('\n')[0]
        sep = detect_separator(first_line)
        
        # 2. Chargement Pandas (Tout en string pour éviter les erreurs initiales)
        df = pd.read_csv(io.StringIO(content), sep=sep, dtype=str)
        
        # 3. Standardisation des colonnes (Le point critique)
        df = standardize_columns(df)
        
        # Vérification des colonnes minimales requises
        required = ['CompteNum', 'Debit', 'Credit']
        missing = [c for c in required if c not in df.columns]
        
        if missing:
            st.error(f"Colonnes manquantes dans {uploaded_file.name}. Colonnes trouvées : {list(df.columns)}. Manquantes : {missing}")
            # Fallback : Si on ne trouve pas Debit/Credit, on cherche des colonnes contenant "Debit"
            return None

        # 4. Nettoyage des Montants
        df['MontantDebit'] = clean_financial_number(df['Debit'])
        df['MontantCredit'] = clean_financial_number(df['Credit'])
        
        # 5. Nettoyage des Comptes (garder que les chiffres)
        df['CompteNum'] = df['CompteNum'].astype(str).str.replace(r'\D', '', regex=True) # Garde que 0-9

        # 6. Gestion des Dates (Souvent la cause du problème)
        if 'EcritureDate' in df.columns:
            # On essaie d'abord le format standard FEC (YYYYMMDD)
            df['Date_Analyse'] = pd.to_datetime(df['EcritureDate'], format='%Y%m%d', errors='coerce')
            
            # Si ça a échoué (NaT), on essaie le format Excel classique (DD/MM/YYYY)
            mask_nat = df['Date_Analyse'].isna()
            if mask_nat.any():
                df.loc[mask_nat, 'Date_Analyse'] = pd.to_datetime(df.loc[mask_nat, 'EcritureDate'], dayfirst=True, errors='coerce')
            
            # On supprime les lignes sans date valide
            df = df.dropna(subset=['Date_Analyse'])
        else:
            st.error(f"Pas de colonne de Date trouvée dans {uploaded_file.name}")
            return None

        return df

    except Exception as e:
        st.error(f"Erreur technique lecture {uploaded_file.name}: {str(e)}")
        return None

# --- 2. CALCUL DES INDICATEURS (VOTRE LOGIQUE) ---

def calculer_indicateurs_mensuels(df):
    if df.empty: return pd.DataFrame()
    
    df = df.set_index('Date_Analyse').sort_index()
    groupe_mois = df.groupby(pd.Grouper(freq='ME'))
    resultats = []

    for mois, data in groupe_mois:
        if data.empty: continue

        # On utilise les colonnes nettoyées MontantDebit / MontantCredit
        
        # CA (70...)
        mask_ca = data['CompteNum'].str.startswith('70')
        ca = (data.loc[mask_ca, 'MontantCredit'] - data.loc[mask_ca, 'MontantDebit']).sum()

        # EBITDA
        mask_prod = data['CompteNum'].str.match(r'^(70|71|72|73|74)')
        prod = (data.loc[mask_prod, 'MontantCredit'] - data.loc[mask_prod, 'MontantDebit']).sum()
        
        mask_chg = data['CompteNum'].str.match(r'^(60|61|62|63|64)')
        chg = (data.loc[mask_chg, 'MontantDebit'] - data.loc[mask_chg, 'MontantCredit']).sum()
        
        ebitda = prod - chg

        # Résultat Net
        mask_cl7 = data['CompteNum'].str.startswith('7')
        total_prod = (data.loc[mask_cl7, 'MontantCredit'] - data.loc[mask_cl7, 'MontantDebit']).sum()
        
        mask_cl6 = data['CompteNum'].str.startswith('6')
        total_chg = (data.loc[mask_cl6, 'MontantDebit'] - data.loc[mask_cl6, 'MontantCredit']).sum()
        
        resultat = total_prod - total_chg

        # Tréso
        mask_treso = data['CompteNum'].str.startswith('5')
        flux_treso = (data.loc[mask_treso, 'MontantDebit'] - data.loc[mask_treso, 'MontantCredit']).sum()

        resultats.append({
            'Date': mois,
            'CA': ca,
            'EBITDA': ebitda,
            'Resultat': resultat,
            'Flux_Treso': flux_treso
        })

    df_res = pd.DataFrame(resultats)
    if not df_res.empty:
        df_res = df_res.set_index('Date')
        df_res['Treso_Cumulee'] = df_res['Flux_Treso'].cumsum()
    
    return df_res

# --- 3. PRÉDICTIONS (MATHS) ---

def generate_predictions(df_history, months_to_predict):
    if df_history.empty or len(df_history) < 2:
        return None

    df = df_history.copy()
    df['Time_Index'] = np.arange(len(df))
    
    last_date = df.index[-1]
    future_dates = pd.date_range(start=last_date, periods=months_to_predict + 1, freq='ME')[1:]
    
    predictions = pd.DataFrame(index=future_dates)
    future_time_index = np.arange(len(df), len(df) + months_to_predict).reshape(-1, 1)

    indicators = ['CA', 'EBITDA', 'Resultat', 'Treso_Cumulee']
    
    for col in indicators:
        X = df[['Time_Index']]
        y = df[col].fillna(0)
        
        model = LinearRegression()
        model.fit(X, y)
        trend_future = model.predict(future_time_index)
        predictions[col] = trend_future

    return predictions

# --- 4. INTERFACE ---

# Sidebar
st.sidebar.header("Paramètres")
api_key = st.sidebar.text_input("Clé API Gemini", type="password")
uploaded_files = st.sidebar.file_uploader("Fichiers FEC (.txt, .csv)", accept_multiple_files=True)
horizon_years = st.sidebar.slider("Horizon prédiction (années)", 1, 3, 1)

st.title("📊 Dashboard Financier Robustifié")

if uploaded_files:
    all_dfs = []
    
    # --- SECTION DEBUGGING ---
    debug_expander = st.expander("🕵️‍♂️ Voir les données brutes (Debug)", expanded=False)
    
    for file in uploaded_files:
        df_clean = load_fec_robust(file)
        if df_clean is not None:
            all_dfs.append(df_clean)
            # Affichage debug pour le premier fichier
            with debug_expander:
                st.write(f"**Fichier : {file.name}**")
                st.write(f"Colonnes détectées : {list(df_clean.columns)}")
                st.dataframe(df_clean.head(5))
                st.write(f"Nombre de lignes valides : {len(df_clean)}")
            
    if all_dfs:
        df_global = pd.concat(all_dfs, ignore_index=True)
        
        # Calcul
        df_history = calculer_indicateurs_mensuels(df_global)
        
        if not df_history.empty:
            months_pred = horizon_years * 12
            df_pred = generate_predictions(df_history, months_pred)
            
            # Dernier mois
            last_month = df_history.iloc[-1]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Dernier CA Mensuel", f"{last_month['CA']:,.0f} €")
            c2.metric("Dernier EBITDA", f"{last_month['EBITDA']:,.0f} €")
            c3.metric("Dernier Résultat", f"{last_month['Resultat']:,.0f} €")
            c4.metric("Trésorerie Actuelle", f"{last_month['Treso_Cumulee']:,.0f} €")

            st.markdown("---")

            # Graphiques
            indicators_config = [
                {'col': 'CA', 'title': "Chiffre d'Affaires", 'color': '#1f77b4'},
                {'col': 'EBITDA', 'title': "EBITDA", 'color': '#ff7f0e'},
                {'col': 'Resultat', 'title': "Résultat Net", 'color': '#2ca02c'},
                {'col': 'Treso_Cumulee', 'title': "Trésorerie Cumulée", 'color': '#9467bd'}
            ]

            col_layout = st.columns(2)
            
            for i, config in enumerate(indicators_config):
                col_idx = i % 2
                with col_layout[col_idx]:
                    fig = go.Figure()

                    # Historique
                    fig.add_trace(go.Scatter(
                        x=df_history.index, 
                        y=df_history[config['col']],
                        mode='lines',
                        name='Historique',
                        line=dict(color=config['color'], width=3)
                    ))

                    # Prédiction
                    if df_pred is not None:
                        last_point = df_history.iloc[[-1]][[config['col']]]
                        pred_combined = pd.concat([last_point, df_pred[[config['col']]]])
                        
                        fig.add_trace(go.Scatter(
                            x=pred_combined.index,
                            y=pred_combined[config['col']],
                            mode='lines',
                            name='Prédiction',
                            line=dict(color=config['color'], width=3, dash='dot')
                        ))

                    fig.update_layout(title=config['title'], height=350)
                    st.plotly_chart(fig, use_container_width=True)

        else:
            st.error("❌ Les données ont été chargées mais le calcul a donné 0 résultats. Vérifiez que la colonne 'CompteNum' contient bien des comptes de classe 6 et 7 dans la section Debug ci-dessus.")
