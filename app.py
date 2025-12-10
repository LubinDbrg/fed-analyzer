import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import io
import google.generativeai as genai
import re

# Configuration de la page
st.set_page_config(page_title="KPI Restauration - STC", layout="wide")

# Noms des fichiers (Assurez-vous qu'ils sont dans le même dossier)
LOGO_FILE = "593989620_2217848631958856_7080388737174534799_n.png"
ADVICE_FILE = "conseil_entreprises.txt"

# --- 0. CONFIGURATION & DONNÉES ---
EVENTS_DB = [
    {"date": "2021-05-19", "label": "Terrasses", "color": "#FFFF00"},
    {"date": "2021-06-09", "label": "Salles", "color": "#FFFF00"},
    {"date": "2021-08-09", "label": "Pass Sanitaire", "color": "#FFAE00"},
    {"date": "2022-02-24", "label": "Guerre Ukraine", "color": "#FF0000"},
    {"date": "2022-07-25", "label": "Canicule", "color": "#FF5733"},
    {"date": "2023-09-08", "label": "Rugby WC", "color": "#33FF57"},
    {"date": "2023-10-07", "label": "Guerre Gaza", "color": "#FF0000"},
    {"date": "2024-07-26", "label": "JO Paris", "color": "#33A1FF"},
    {"date": "2025-01-20", "label": "Tarifs Trump", "color": "#800080"},
]

# --- 1. FONCTIONS UTILITAIRES ---

def clear_fec_cache():
    if "fec_uploader" in st.session_state:
        st.session_state["fec_uploader"] = []

def format_fr_currency(value):
    """Formate un nombre avec un POINT pour les milliers et ajoute €."""
    us_fmt = f"{value:,.0f}"
    fr_fmt = us_fmt.replace(',', '.')
    return f"{fr_fmt} €"

def add_context_to_figure(fig, start_date, end_date):
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    for event in EVENTS_DB:
        event_date = pd.Timestamp(event["date"])
        if start_ts <= event_date <= end_ts:
            fig.add_vline(x=event_date, line_width=1, line_dash="dot", line_color=event["color"])
            fig.add_annotation(
                x=event_date, y=1.05, yref="paper", text=event["label"],
                showarrow=False, font=dict(size=10, color=event["color"]), textangle=-90
            )
    return fig

@st.dialog("Analyse Détaillée & Contextuelle", width="large")
def show_zoomed_chart(fig_base, title, start_date, end_date):
    st.subheader(f"🔎 Zoom : {title}")
    fig_zoom = go.Figure(fig_base)
    fig_zoom = add_context_to_figure(fig_zoom, start_date, end_date)
    fig_zoom.update_layout(height=600, title=f"{title} (avec évènements majeurs)", margin=dict(t=80))
    st.plotly_chart(fig_zoom, use_container_width=True)
    st.info("Les lignes pointillées représentent les évènements extra-financiers.")

# --- 2. MOTEURS DE CONSEILS (IA & FICHIER TXT) ---

# --- MOTEUR 1 : IA GEMINI ---
def get_best_available_model():
    try:
        available_models = []
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                available_models.append(m.name)
        for m in available_models:
            if "gemini-1.5-flash" in m and "exp" not in m: return m
        for m in available_models:
            if "gemini-1.5-pro" in m and "exp" not in m: return m
        for m in available_models:
            if "flash" in m: return m
        return available_models[0] if available_models else None
    except: return None

def generer_conseils_gemini(api_key, stats_dict, scenario_nom):
    if not api_key: return "⚠️ Veuillez entrer votre clé API Gemini dans la barre latérale."
    try:
        genai.configure(api_key=api_key)
        model_name = get_best_available_model()
        if not model_name: return "❌ Aucun modèle compatible trouvé sur cette clé API."

        prompt = f"""
        Tu es un expert CFO spécialisé dans le secteur de la restauration.
        Analyse la situation financière ACTUELLE (Dernier mois clôturé) :
        - CA : {stats_dict['CA']}
        - EBITDA : {stats_dict['EBITDA']}
        - Résultat Net : {stats_dict['Resultat']}
        - Trésorerie : {stats_dict['Treso']}
        
        CONTEXTE PRÉVISIONNEL : Scénario {scenario_nom}
        
        MISSION : Donne 3 conseils stratégiques et opérationnels précis (Marge, Staff, Cash) pour améliorer ces résultats actuels.
        """
        model = genai.GenerativeModel(model_name)
        with st.spinner(f'🤖 Analyse IA en cours ({model_name})...'):
            response = model.generate_content(prompt)
            return response.text
    except Exception as e:
        if "429" in str(e): return "⚠️ Limite de quota IA atteinte. Réessayez dans une minute."
        return f"❌ Erreur technique IA : {e}"

# --- MOTEUR 2 : LECTURE DU FICHIER TXT (PRÉ-GÉNÉRÉ) ---
@st.cache_data
def load_advice_database(filepath):
    """Lit le fichier texte et crée un dictionnaire {ID_DOSSIER: TEXTE_COMPLET}"""
    advice_db = {}
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        
        # On découpe par "DOSSIER :"
        # Le regex cherche "DOSSIER :" suivi de l'ID
        sections = re.split(r'(DOSSIER\s*:\s*)', content)
        
        current_id = None
        
        for i in range(1, len(sections), 2):
            header_marker = sections[i] # "DOSSIER : "
            body = sections[i+1] # Le reste du texte jusqu'au prochain split
            
            # Extraction de l'ID (ex: "000003" dans "000003  (Exercice 2025)...")
            # On prend la première ligne du body
            first_line = body.strip().split('\n')[0]
            # On prend le premier mot de cette ligne comme ID
            dossier_id = first_line.split()[0].strip()
            
            # On reconstruit le texte complet
            full_text = header_marker + body
            
            # Nettoyage des séparateurs de fin si besoin
            full_text = full_text.split("...................")[0]
            
            advice_db[dossier_id] = full_text.strip()
            
        return advice_db
    except Exception as e:
        st.error(f"Erreur de lecture du fichier conseils : {e}")
        return {}

def get_advice_from_txt(uploaded_filename, advice_db):
    """Cherche le conseil correspondant au fichier uploadé"""
    if not advice_db:
        return "❌ Base de données de conseils vide ou introuvable."
    
    # On essaie de trouver l'ID du dossier dans le nom du fichier
    # Ex: Si fichier = "FEC_000003_2025.csv", on cherche si "000003" est une clé
    found_key = None
    
    # Méthode 1 : Correspondance exacte ou partielle
    for db_id in advice_db.keys():
        if db_id in uploaded_filename:
            found_key = db_id
            break
            
    if found_key:
        return f"### 📄 Rapport Dossier {found_key}\n\n```text\n{advice_db[found_key]}\n```"
    else:
        return f"⚠️ Aucun rapport pré-généré trouvé pour le fichier : `{uploaded_filename}`.\n\nAssurez-vous que l'ID du dossier (ex: 000003) est présent dans le nom du fichier."

# --- 3. FONCTIONS DE LECTURE & CALCULS ---
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

# --- 4. PRÉDICTION HYBRIDE ---
def create_features(df, label=None):
    df = df.copy()
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    df['dayofyear'] = df.index.dayofyear
    df['time_idx'] = np.arange(len(df))
    return df

def predict_hybrid_ca(series, months_to_predict, trend_factor=1.0):
    series = series.fillna(0)
    if len(series) < 2: return None
    try:
        df = pd.DataFrame({'y': series})
        df = create_features(df)
        X = df[['time_idx', 'month', 'quarter']]
        y = df['y']
        model_trend = LinearRegression()
        model_trend.fit(df[['time_idx']], y)
        trend_pred = model_trend.predict(df[['time_idx']])
        y_residuals = y - trend_pred
        model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
        model_rf.fit(X, y_residuals)
        last_date = series.index[-1]
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
        growth_curve = np.linspace(1, trend_factor, len(final_pred))
        final_pred = final_pred * growth_curve
        final_pred = np.maximum(final_pred, 0)
        return pd.Series(final_pred, index=future_dates)
    except Exception as e:
        print(f"Erreur prédiction: {e}")
        return None

# --- 5. INTERFACE PRINCIPALE ---

# AJOUT DU LOGO EN HAUT (NOUVEAU FICHIER)
try:
    st.image(LOGO_FILE, width=300)
except:
    st.warning(f"Image '{LOGO_FILE}' introuvable.")

st.title("📊 Finance & Restauration : Dashboard Hybride")

st.sidebar.header("Paramètres & Données")
col_suppr, col_upload = st.sidebar.columns([0.2, 0.8])
st.sidebar.button("🗑️", on_click=clear_fec_cache, help="Efface toutes les données chargées")
uploaded_files = st.sidebar.file_uploader("Fichiers FEC (Glisser-déposer)", accept_multiple_files=True, key="fec_uploader")
horizon_years = st.sidebar.slider("Horizon prédiction (années)", 1, 3, 2)

st.sidebar.markdown("---")
st.sidebar.subheader("🧠 Méthode de Conseil")

# CHOIX DE LA MÉTHODE DE CONSEIL (MODIFIÉ)
method_choice = st.sidebar.radio(
    "Choisir le moteur d'analyse :",
    ("🤖 IA Générative (Gemini)", "📄 Rapport Pré-généré (Fichier TXT)"),
    index=0
)

if method_choice == "🤖 IA Générative (Gemini)":
    api_key_input = st.sidebar.text_input("Clé API Gemini", type="password")
else:
    api_key_input = None 
    # Pré-chargement de la base de conseils
    advice_db = load_advice_database(ADVICE_FILE)

st.sidebar.subheader("🌍 Scénario Éco (Pour Prévisions)")
scenario_map = {
    "Neutre": 1.0, "Optimiste (+5%)": 1.05, "Pessimiste (-5%)": 0.95,
    "Inflation (+1.5%)": 1.015
}
choix_scenario = st.sidebar.selectbox("Tendance :", list(scenario_map.keys()))
trend_factor = scenario_map[choix_scenario]


if uploaded_files:
    all_dfs = []
    # On garde le nom du premier fichier pour la recherche de conseils
    first_filename = uploaded_files[0].name 
    
    for file in uploaded_files:
        df = load_fec_robust(file)
        if df is not None: all_dfs.append(df)

    if all_dfs:
        df_global = pd.concat(all_dfs, ignore_index=True)
        df_mensuel = calculer_indicateurs_mensuels(df_global)
        serie_treso_jour = calculer_tresorerie_quotidienne(df_global)

        if not df_mensuel.empty:
            months_pred = horizon_years * 12
            
            global_min_date = df_mensuel.index.min()
            global_max_date = df_mensuel.index.max() + pd.DateOffset(months=months_pred)

            last_m = df_mensuel.iloc[-1]
            last_treso = serie_treso_jour.iloc[-1] if not serie_treso_jour.empty else 0
            
            # --- KPI Cards (TITRES MODIFIÉS) ---
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("📅 CA Mensuel (Dernier Mois)", format_fr_currency(last_m['CA']))
            c2.metric("⚡ EBITDA (Dernier Mois)", format_fr_currency(last_m['EBITDA']))
            c3.metric("💰 Résultat Net (Dernier Mois)", format_fr_currency(last_m['Resultat']))
            c4.metric("🏦 Trésorerie (Actuelle)", format_fr_currency(last_treso))
            
            st.markdown("---")

            # --- CALCUL PRÉDICTIONS ---
            with st.spinner('Calcul des modèles prédictifs...'):
                pred_ca = predict_hybrid_ca(df_mensuel['CA'], months_pred, trend_factor)
            
            pred_ebitda = None
            pred_result = None
            pred_treso = None

            if pred_ca is not None:
                last_12 = df_mensuel.iloc[-12:] if len(df_mensuel) >= 12 else df_mensuel
                sum_ca = last_12['CA'].sum()
                marge_ebitda = (last_12['EBITDA'].sum() / sum_ca) if sum_ca != 0 else 0
                pred_ebitda = pred_ca * marge_ebitda
                ecart_resultat = (last_12['EBITDA'] - last_12['Resultat']).mean()
                pred_result = pred_ebitda - ecart_resultat

                pred_treso_list = []
                current_cash = last_treso
                for res in pred_result:
                    current_cash += res
                    pred_treso_list.append(current_cash)
                pred_treso = pd.Series(pred_treso_list, index=pred_ca.index)
            else:
                st.warning("⚠️ Historique insuffisant pour prédiction.")

            # --- PREPARATION DES GRAPHES ---
            fig_ca = go.Figure()
            fig_ca.add_trace(go.Bar(x=df_mensuel.index, y=df_mensuel['CA'], name='Historique', marker_color='#1f77b4'))
            if pred_ca is not None:
                fig_ca.add_trace(go.Bar(x=pred_ca.index, y=pred_ca, name='Prévision AI', marker_pattern_shape='/', marker_color='#4ad3d8', opacity=0.7))
            fig_ca.update_layout(title="Chiffre d'Affaires", height=350, template="plotly_dark", margin=dict(l=20, r=20, t=40, b=20))

            fig_eb = go.Figure()
            fig_eb.add_trace(go.Scatter(x=df_mensuel.index, y=df_mensuel['EBITDA'], mode='lines+markers', name='Historique', line=dict(color='#ff7f0e', width=3)))
            if pred_ebitda is not None:
                fig_eb.add_trace(go.Scatter(x=pred_ebitda.index, y=pred_ebitda, mode='lines+markers', name='Prévision', line=dict(color='#00CC96', width=4, dash='dash'), marker=dict(size=8, symbol='diamond')))
            fig_eb.add_hline(y=0, line_color="white", line_width=1, opacity=0.3)
            fig_eb.update_layout(title="EBITDA (Rentabilité)", height=350, template="plotly_dark", margin=dict(l=20, r=20, t=40, b=20))

            fig_res = go.Figure()
            colors_hist = ['#2ca02c' if v >= 0 else '#d62728' for v in df_mensuel['Resultat']]
            fig_res.add_trace(go.Bar(x=df_mensuel.index, y=df_mensuel['Resultat'], name='Historique', marker_color=colors_hist))
            if pred_result is not None:
                colors_pred = ['#5cd65c' if v >= 0 else '#ff6666' for v in pred_result]
                fig_res.add_trace(go.Bar(x=pred_result.index, y=pred_result, name='Prévision', marker_pattern_shape='x', marker_color=colors_pred, opacity=0.8))
            fig_res.add_hline(y=0, line_color="white", line_width=1, opacity=0.3)
            fig_res.update_layout(title="Résultat Net", height=350, template="plotly_dark", margin=dict(l=20, r=20, t=40, b=20))

            fig_tr = go.Figure()
            treso_lisse = serie_treso_jour.resample('ME').last()
            fig_tr.add_trace(go.Scatter(x=treso_lisse.index, y=treso_lisse, mode='lines', name='Historique', fill='tozeroy', line=dict(color='#9467bd', width=2)))
            if pred_treso is not None:
                fig_tr.add_trace(go.Scatter(x=pred_treso.index, y=pred_treso, mode='lines', name='Prévision', fill='tonexty', line=dict(color='#D670D6', width=3, dash='dash')))
            fig_tr.add_hline(y=0, line_color="red", line_width=1, line_dash="dot")
            fig_tr.update_layout(title="Trésorerie", height=350, template="plotly_dark", margin=dict(l=20, r=20, t=40, b=20))

            # --- AFFICHAGE GRID AVEC BOUTONS ZOOM ---
            col1, col2 = st.columns(2)
            with col1:
                c1_head, c1_btn = st.columns([0.8, 0.2])
                c1_head.write("")
                if c1_btn.button("🔍 Agrandir", key="btn_ca"): show_zoomed_chart(fig_ca, "Chiffre d'Affaires", global_min_date, global_max_date)
                st.plotly_chart(fig_ca, use_container_width=True)
            with col2:
                c2_head, c2_btn = st.columns([0.8, 0.2])
                c2_head.write("")
                if c2_btn.button("🔍 Agrandir", key="btn_ebitda"): show_zoomed_chart(fig_eb, "EBITDA", global_min_date, global_max_date)
                st.plotly_chart(fig_eb, use_container_width=True)

            col3, col4 = st.columns(2)
            with col3:
                c3_head, c3_btn = st.columns([0.8, 0.2])
                c3_head.write("")
                if c3_btn.button("🔍 Agrandir", key="btn_res"): show_zoomed_chart(fig_res, "Résultat Net", global_min_date, global_max_date)
                st.plotly_chart(fig_res, use_container_width=True)
            with col4:
                c4_head, c4_btn = st.columns([0.8, 0.2])
                c4_head.write("")
                if c4_btn.button("🔍 Agrandir", key="btn_tr"): show_zoomed_chart(fig_tr, "Trésorerie", global_min_date, global_max_date)
                st.plotly_chart(fig_tr, use_container_width=True)
            
            # --- SECTION CONSEILS (MÉTHODE AU CHOIX) ---
            st.markdown("---")
            st.subheader(f"👨‍🍳 Conseils Stratégiques : Méthode {method_choice}")
            
            col_ai_btn, col_ai_txt = st.columns([0.2, 0.8])
            
            if col_ai_btn.button("🚀 Générer l'analyse"):
                
                # CAS 1 : Méthode IA Gemini
                if method_choice == "🤖 IA Générative (Gemini)":
                    if api_key_input:
                        stats_gemini = {
                            'CA': format_fr_currency(last_m['CA']),
                            'EBITDA': format_fr_currency(last_m['EBITDA']),
                            'Resultat': format_fr_currency(last_m['Resultat']),
                            'Treso': format_fr_currency(last_treso)
                        }
                        conseils = generer_conseils_gemini(api_key_input, stats_gemini, choix_scenario)
                        if "❌" in conseils or "⚠️" in conseils:
                            st.warning(conseils)
                        else:
                            st.success("Analyse IA générée !")
                            st.markdown(conseils)
                    else:
                        st.error("Veuillez entrer une clé API Gemini pour utiliser l'IA.")

                # CAS 2 : Lecture Fichier TXT (NOUVEAU)
                elif method_choice == "📄 Rapport Pré-généré (Fichier TXT)":
                    # On cherche le conseil correspondant au nom du fichier uploadé
                    rapport_txt = get_advice_from_txt(first_filename, advice_db)
                    st.success("Rapport chargé !")
                    st.markdown(rapport_txt)
            
            else:
                st.info(f"Cliquez pour lancer l'analyse avec la méthode : {method_choice}")
