import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import io
import google.generativeai as genai
import re
import os
import json
import zipfile

# Configuration de la page
st.set_page_config(page_title="KPI Restauration - STC", layout="wide")

# --- CONFIGURATION FICHIERS & URL ---
LOGO_URL = "https://scontent-mrs2-3.xx.fbcdn.net/v/t1.15752-9/593989620_2217848631958856_7080388737174534799_n.png?stp=dst-png_p394x394&_nc_cat=104&ccb=1-7&_nc_sid=0024fc&_nc_ohc=Rkmg6RI2seYQ7kNvwHe0B0i&_nc_oc=AdlAv8BhqxR27G2lrlER10hKoJbxWWIaOYh_MoFdUMTGRD1co3jYPzFyucWERnVzeHM&_nc_ad=z-m&_nc_cid=0&_nc_zt=23&_nc_ht=scontent-mrs2-3.xx&oh=03_Q7cD4AFbWq-h_BtPiU15YRrwm39u0HJtb-bB6ujQEuOrM6JIpQ&oe=6960DB09"

ADVICE_FILE = "conseil_entreprises.txt"
PROFILE_FILE = "rapport_profils_detaille.txt"
DATA_ROOT_DIR = "FEC_site"

# --- 0. DONNÉES ---
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
    {"date": "2026-06-11", "label": "Mondial Foot 26", "color": "#33FF57"},
    {"date": "2027-04-10", "label": "Présidentielle FR", "color": "#0000FF"},
    {"date": "2028-07-14", "label": "JO Los Angeles", "color": "#33A1FF"},
]

SCENARIOS = {
    "Neutre": {"factor": 1.0, "color": "#33C1FF"},
    "Optimiste": {"factor": 1.05, "color": "#00E676"},
    "Pessimiste": {"factor": 0.95, "color": "#FF5252"},
    "Inflation": {"factor": 1.015, "color": "#FFD700"}
}

# --- 1. GESTION SESSION STATE ---
if 'forecast_data' not in st.session_state:
    st.session_state['forecast_data'] = {}
if 'advice_result' not in st.session_state:
    st.session_state['advice_result'] = None
if 'last_company' not in st.session_state:
    st.session_state['last_company'] = None

# --- 2. FONCTIONS UTILITAIRES ---

def format_fr_currency(value):
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

def clear_fec_cache():
    st.session_state["fec_uploader"] = None
    st.session_state['forecast_data'] = {} 
    st.session_state['advice_result'] = None

@st.dialog("Analyse Détaillée & Contextuelle", width="large")
def show_zoomed_chart(fig_base, title, start_date, end_date):
    st.subheader(f"🔎 Zoom : {title}")
    fig_zoom = go.Figure(fig_base)
    fig_zoom = add_context_to_figure(fig_zoom, start_date, end_date)
    fig_zoom.update_layout(height=600, title=f"{title} (avec évènements majeurs)", margin=dict(t=80))
    st.plotly_chart(fig_zoom, use_container_width=True)
    st.info("Les lignes pointillées représentent les évènements extra-financiers.")

# --- 3. MOTEURS LLM (GEMINI) ---

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

def generer_conseils_gemini(api_key, stats_dict):
    if not api_key: return "⚠️ Veuillez entrer votre clé API Gemini."
    try:
        genai.configure(api_key=api_key)
        model_name = get_best_available_model()
        if not model_name: return "❌ Aucun modèle compatible trouvé."

        prompt = f"""
        Tu es un expert CFO en restauration.
        Situation Actuelle : CA={stats_dict['CA']}, EBITDA={stats_dict['EBITDA']}, Résultat={stats_dict['Resultat']}, Trésorerie={stats_dict['Treso']}.
        Mission : Donne 3 conseils stratégiques (Marge, Staff, Cash) pour améliorer la situation.
        """
        model = genai.GenerativeModel(model_name)
        with st.spinner(f'🤖 Rédaction des conseils ({model_name})...'):
            response = model.generate_content(prompt)
            return response.text
    except Exception as e:
        return f"❌ Erreur IA : {e}"

def predict_gemini_forecasting(series_history, months_to_predict, trend_factor, api_key, scenario_name):
    if not api_key: return None
    try:
        genai.configure(api_key=api_key)
        model_name = get_best_available_model()
        model = genai.GenerativeModel(model_name)
        
        history_str = ", ".join([str(int(x)) for x in series_history.values])
        last_date = series_history.index[-1]
        prediction_end_date = last_date + pd.DateOffset(months=months_to_predict)
        
        relevant_events = []
        for evt in EVENTS_DB:
            evt_date = pd.Timestamp(evt["date"])
            if last_date < evt_date <= prediction_end_date:
                relevant_events.append(f"- {evt['date']}: {evt['label']}")
        
        future_events_str = "Prends en compte ces évènements :" + "\n".join(relevant_events) if relevant_events else ""

        prompt = f"""
        Tu es un expert en prévision financière.
        HISTORIQUE CA : [{history_str}]
        SCÉNARIO : {scenario_name} (Facteur tendance: {trend_factor})
        MISSION : Prédire le CA pour les {months_to_predict} prochains mois.
        RÉPONSE : UNIQUEMENT une liste JSON d'entiers. Ex: [12000, 14500]
        """
        
        response = model.generate_content(prompt)
        text_resp = response.text
        
        json_match = re.search(r'\[.*\]', text_resp, re.DOTALL)
        if json_match:
            clean_json = json_match.group(0)
            predicted_values = json.loads(clean_json)
            if len(predicted_values) > months_to_predict:
                predicted_values = predicted_values[:months_to_predict]
            elif len(predicted_values) < months_to_predict:
                avg = sum(predicted_values)/len(predicted_values)
                predicted_values.extend([avg] * (months_to_predict - len(predicted_values)))
            
            future_dates = pd.date_range(start=last_date, periods=months_to_predict + 1, freq='ME')[1:]
            return pd.Series(predicted_values, index=future_dates)
        else:
            return None
    except Exception as e:
        print(f"Erreur Gemini Forecasting: {e}")
        return None

# --- 4. LOGIQUE FICHIER TEXTE (PROFILS & CONSEILS) ---

@st.cache_data
def load_text_database(filepath):
    """Charge un fichier texte structuré par 'DOSSIER :'"""
    db = {}
    try:
        if not os.path.exists(filepath): return {}
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        
        # On découpe par le marqueur "DOSSIER :"
        sections = re.split(r'(DOSSIER\s*:\s*)', content)
        
        for i in range(1, len(sections), 2):
            body = sections[i+1]
            
            # Extraction propre de l'ID (première ligne, premier mot)
            first_line = body.strip().split('\n')[0]
            # On nettoie pour ne garder que le code (ex: "000003")
            dossier_id = first_line.split()[0].strip()
            
            full_text = sections[i] + body
            # Nettoyage des séparateurs de fin
            db[dossier_id] = full_text.split("...................")[0].strip()
            
        return db
    except Exception as e:
        # st.error(f"Erreur lecture fichier {filepath}: {e}")
        return {}

def get_text_from_db(company_folder_name, db):
    if not db: return None
    
    # 1. Correspondance Exacte
    if company_folder_name in db:
        return db[company_folder_name]
        
    # 2. Correspondance Partielle (si le dossier est "000003_NOM" et la clé "000003")
    for key in db.keys():
        if key in company_folder_name or company_folder_name in key:
            return db[key]
            
    return None

# --- 5. FONCTIONS LECTURE & MATHS ---

def detect_separator(line):
    return ';' if line.count(';') > line.count(',') else ','

def clean_financial_number(series):
    return pd.to_numeric(series.astype(str).str.replace(r'\s+', '', regex=True).str.replace(',', '.'), errors='coerce').fillna(0.0)

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
                final_rename[col] = standard; break
    if final_rename: df = df.rename(columns=final_rename)
    return df

def load_fec_robust(file_path):
    try:
        with open(file_path, 'rb') as f: content_bytes = f.read()
        try: content = content_bytes.decode('latin-1')
        except: content = content_bytes.decode('utf-8', errors='ignore')
        sep = detect_separator(content.split('\n')[0])
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
    flux = df_treso['Flux'].resample('D').sum().fillna(0)
    return flux.cumsum()

# --- ALGO HYBRIDE ---
def create_features(df):
    df = df.copy()
    df['month'] = df.index.month
    df['quarter'] = df.index.quarter
    df['time_idx'] = np.arange(len(df))
    return df

def predict_hybrid_ca(series, months_to_predict, trend_factor):
    if len(series) < 2: return None
    try:
        df = pd.DataFrame({'y': series})
        df = create_features(df)
        X, y = df[['time_idx', 'month', 'quarter']], df['y']
        model_trend = LinearRegression()
        model_trend.fit(df[['time_idx']], y)
        trend_pred = model_trend.predict(df[['time_idx']])
        model_rf = RandomForestRegressor(n_estimators=100)
        model_rf.fit(X, y - trend_pred)
        last_date = series.index[-1]
        future_dates = pd.date_range(start=last_date, periods=months_to_predict + 1, freq='ME')[1:]
        future_df = pd.DataFrame(index=future_dates)
        future_df['month'] = future_df.index.month
        future_df['quarter'] = future_df.index.quarter
        future_df['time_idx'] = np.arange(len(df), len(df) + months_to_predict)
        future_trend = model_trend.predict(future_df[['time_idx']])
        future_residuals = model_rf.predict(future_df[['time_idx', 'month', 'quarter']])
        final_pred = (future_trend + future_residuals) * np.linspace(1, trend_factor, len(future_trend))
        return pd.Series(np.maximum(final_pred, 0), index=future_dates)
    except: return None

# --- 6. INTERFACE PRINCIPALE ---

st.image(LOGO_URL, width=300)
st.title("📊 Finance & Restauration : Dashboard Hybride")

# --- BARRE LATÉRALE ---
st.sidebar.header("📂 Sélection Entreprise")

companies = []
if os.path.exists(DATA_ROOT_DIR) and os.path.isdir(DATA_ROOT_DIR):
    companies = sorted([d for d in os.listdir(DATA_ROOT_DIR) if os.path.isdir(os.path.join(DATA_ROOT_DIR, d))])

selected_company = None
all_dfs = []

if companies:
    selected_company = st.sidebar.selectbox("🏢 Entreprise :", companies)
    
    # RESET CACHE SI CHANGEMENT ENTREPRISE
    if selected_company != st.session_state['last_company']:
        st.session_state['forecast_data'] = {}
        st.session_state['advice_result'] = None
        st.session_state['last_company'] = selected_company

    if selected_company:
        folder_path = os.path.join(DATA_ROOT_DIR, selected_company)
        files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.csv', '.txt'))]
        if files:
            with st.spinner("Chargement des FEC..."):
                for f in files:
                    df = load_fec_robust(os.path.join(folder_path, f))
                    if df is not None: all_dfs.append(df)
            if all_dfs: st.sidebar.success(f"✅ {len(all_dfs)} fichiers chargés.")
        else: st.sidebar.warning("Dossier vide.")
else:
    st.sidebar.error(f"Dossier '{DATA_ROOT_DIR}' introuvable.")

# 2. Paramètres Prévision
st.sidebar.markdown("---")
st.sidebar.subheader("📈 Prévisions & Conseils")

# CHOIX METHODE PREVISION
forecast_method = st.sidebar.radio(
    "Méthode de Prévision (Courbes) :",
    ("Algorithme Hybride (Maths)", "API Gemini (IA + Evènements)"),
    index=0
)

# CHOIX METHODE CONSEIL
advice_method = st.sidebar.radio(
    "Méthode de Conseil (Texte) :",
    ("IA Générative (Gemini)", "Rapport Pré-généré (Fichier)"),
    index=1
)

api_key_input = None
if "Gemini" in forecast_method or "Gemini" in advice_method:
    api_key_input = st.sidebar.text_input("🔑 Clé API Gemini", type="password")

# --- BOUTON DE LANCEMENT (CACHE) ---
st.sidebar.markdown("---")
launch_calc = st.sidebar.button("⚡ Lancer la Prédiction (Tous Scénarios)")

# --- DASHBOARD ---

if all_dfs:
    df_global = pd.concat(all_dfs, ignore_index=True)
    df_m = calculer_indicateurs_mensuels(df_global)
    df_treso = calculer_tresorerie_quotidienne(df_global)

    if not df_m.empty:
        # --- 1. AFFICHAGE DU PROFIL (IMMEDIATEMENT EN HAUT) ---
        profil_db = load_text_database(PROFILE_FILE)
        profil_text = get_text_from_db(selected_company, profil_db)
        
        if profil_text:
            with st.expander(f"👤 Profil Stratégique : {selected_company}", expanded=True):
                st.markdown(f"```text\n{profil_text}\n```")
        else:
            st.info("ℹ️ Aucun profil détaillé disponible pour cette entreprise dans le fichier texte.")

        # --- 2. KPI ---
        st.markdown("---")
        months_pred = 6
        last_m = df_m.iloc[-1]
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("CA Mensuel (Dernier)", format_fr_currency(last_m['CA']))
        c2.metric("EBITDA (Dernier)", format_fr_currency(last_m['EBITDA']))
        c3.metric("Résultat Net (Dernier)", format_fr_currency(last_m['Resultat']))
        c4.metric("Trésorerie (Actuelle)", format_fr_currency(df_treso.iloc[-1] if not df_treso.empty else 0))
        
        st.markdown("---")

        # --- LOGIQUE DE CALCUL MULTI-SCENARIOS ---
        if launch_calc:
            with st.spinner(f'Calcul des {len(SCENARIOS)} scénarios en cours ({forecast_method})...'):
                
                scenarios_results = {}
                
                for scenario_name, scenario_params in SCENARIOS.items():
                    trend_factor = scenario_params["factor"]
                    pred_ca = None
                    
                    if "Hybride" in forecast_method:
                        pred_ca = predict_hybrid_ca(df_m['CA'], months_pred, trend_factor)
                    elif "Gemini" in forecast_method:
                        if api_key_input:
                            pred_ca = predict_gemini_forecasting(df_m['CA'], months_pred, trend_factor, api_key_input, scenario_name)
                        else:
                            st.error("⚠️ Clé API requise.")
                            break

                    pred_ebitda, pred_res, pred_treso = None, None, None
                    if pred_ca is not None:
                        last_12 = df_m.iloc[-12:] if len(df_m) >= 12 else df_m
                        sum_ca = last_12['CA'].sum()
                        marge_ebitda = (last_12['EBITDA'].sum() / sum_ca) if sum_ca > 0 else 0
                        ecart_res = (last_12['EBITDA'] - last_12['Resultat']).mean()
                        
                        pred_ebitda = pred_ca * marge_ebitda
                        pred_res = pred_ebitda - ecart_res
                        
                        start_treso = df_treso.iloc[-1] if not df_treso.empty else 0
                        treso_list = []
                        curr = start_treso
                        for r in pred_res:
                            curr += r
                            treso_list.append(curr)
                        pred_treso = pd.Series(treso_list, index=pred_ca.index)
                        
                        scenarios_results[scenario_name] = {
                            'ca': pred_ca, 'ebitda': pred_ebitda, 'res': pred_res, 'treso': pred_treso
                        }

                st.session_state['forecast_data'] = scenarios_results
                
            if "Gemini" in advice_method:
                if api_key_input:
                    stats = {k: format_fr_currency(v) for k,v in {'CA': last_m['CA'], 'EBITDA': last_m['EBITDA'], 'Resultat': last_m['Resultat'], 'Treso': df_treso.iloc[-1]}.items()}
                    st.session_state['advice_result'] = generer_conseils_gemini(api_key_input, stats)
            else:
                advice_db = load_text_database(ADVICE_FILE)
                st.session_state['advice_result'] = get_text_from_db(selected_company, advice_db)

        # --- AFFICHAGE GRAPHIQUES ---
        min_date, max_date = df_m.index.min(), df_m.index.max() + pd.DateOffset(months=months_pred)
        forecasts = st.session_state.get('forecast_data', {})

        col1, col2 = st.columns(2)
        with col1: # CA
            fig = go.Figure()
            fig.add_trace(go.Bar(x=df_m.index, y=df_m['CA'], name='Historique', marker_color='#1f77b4'))
            for s_name, s_data in forecasts.items():
                if s_data['ca'] is not None:
                    fig.add_trace(go.Scatter(x=s_data['ca'].index, y=s_data['ca'], name=f'Scénario {s_name}', line=dict(color=SCENARIOS[s_name]['color'], width=3, dash='dot')))
            fig.update_layout(title="Chiffre d'Affaires", height=350, template="plotly_dark", margin=dict(l=20, r=20, t=40, b=20))
            if st.button("🔍 Zoom CA"): show_zoomed_chart(fig, "CA", min_date, max_date)
            st.plotly_chart(fig, use_container_width=True)

        with col2: # EBITDA
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df_m.index, y=df_m['EBITDA'], mode='lines+markers', name='Historique', line=dict(color='#ff7f0e', width=3)))
            for s_name, s_data in forecasts.items():
                if s_data['ebitda'] is not None:
                    fig.add_trace(go.Scatter(x=s_data['ebitda'].index, y=s_data['ebitda'], name=f'{s_name}', line=dict(color=SCENARIOS[s_name]['color'], width=2, dash='dot')))
            fig.add_hline(y=0, line_color="white", opacity=0.3)
            fig.update_layout(title="EBITDA", height=350, template="plotly_dark", margin=dict(l=20, r=20, t=40, b=20))
            if st.button("🔍 Zoom EBITDA"): show_zoomed_chart(fig, "EBITDA", min_date, max_date)
            st.plotly_chart(fig, use_container_width=True)

        col3, col4 = st.columns(2)
        with col3: # Resultat
            fig = go.Figure()
            colors = ['#2ca02c' if v>=0 else '#d62728' for v in df_m['Resultat']]
            fig.add_trace(go.Bar(x=df_m.index, y=df_m['Resultat'], name='Historique', marker_color=colors))
            for s_name, s_data in forecasts.items():
                if s_data['res'] is not None:
                    fig.add_trace(go.Scatter(x=s_data['res'].index, y=s_data['res'], name=f'{s_name}', line=dict(color=SCENARIOS[s_name]['color'], width=2, dash='dot')))
            fig.update_layout(title="Résultat Net", height=350, template="plotly_dark", margin=dict(l=20, r=20, t=40, b=20))
            if st.button("🔍 Zoom Résultat"): show_zoomed_chart(fig, "Résultat", min_date, max_date)
            st.plotly_chart(fig, use_container_width=True)

        with col4: # Treso
            fig = go.Figure()
            treso_hist_monthly = df_treso.resample('ME').last()
            fig.add_trace(go.Scatter(x=treso_hist_monthly.index, y=treso_hist_monthly, mode='lines', name='Historique', fill='tozeroy', line=dict(color='#9467bd', width=2)))
            for s_name, s_data in forecasts.items():
                if s_data['treso'] is not None:
                    fig.add_trace(go.Scatter(x=s_data['treso'].index, y=s_data['treso'], name=f'{s_name}', fill=None, line=dict(color=SCENARIOS[s_name]['color'], width=3, dash='dot')))
            fig.add_hline(y=0, line_color="red", line_dash="dot")
            fig.update_layout(title="Trésorerie", height=350, template="plotly_dark", margin=dict(l=20, r=20, t=40, b=20))
            if st.button("🔍 Zoom Tréso"): show_zoomed_chart(fig, "Trésorerie", min_date, max_date)
            st.plotly_chart(fig, use_container_width=True)

        # --- AFFICHAGE CONSEILS ---
        st.markdown("---")
        st.subheader(f"👨‍🍳 Conseils Stratégiques ({advice_method})")
        
        if st.session_state.get('advice_result'):
            st.markdown(st.session_state['advice_result'])
        else:
            st.info("Cliquez sur '⚡ Lancer la Prédiction' pour générer le rapport.")

    else:
        st.warning("Données insuffisantes pour l'analyse.")
