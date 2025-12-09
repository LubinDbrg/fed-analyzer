import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import google.generativeai as genai
import io

# --- CONFIGURATION PAGE ---
st.set_page_config(page_title="KPI Finance & Prédictions", layout="wide")


# --- 1. LOGIQUE ISSUE DE VOTRE FICHIER (ADAPTÉE STREAMLIT) ---

def nettoyer_et_preparer_donnees(uploaded_file):
    """
    Logique reprise de 'courbes_KPI.py' adaptée pour lire un buffer mémoire Streamlit.
    """
    try:
        # Lecture du buffer (fichier uploadé)
        # On utilise latin-1 souvent pour les FEC, sinon utf-8
        try:
            content = uploaded_file.getvalue().decode('utf-8')
        except UnicodeDecodeError:
            content = uploaded_file.getvalue().decode('latin-1')

        # Détection basique du séparateur
        sep = ';' if content.count(';') > content.count(',') else ','

        df = pd.read_csv(
            io.StringIO(content),
            sep=sep,
            decimal=',',
            dtype=str  # Important : tout en string au début comme dans votre script
        )

        df.columns = df.columns.str.strip()

        # Normalisation Montants (Logique exacte de votre fichier)
        for col in ['Debit', 'Credit']:
            if col in df.columns:
                df[f'Montant{col}'] = df[col].astype(str).str.replace(',', '.', regex=False)
                df[f'Montant{col}'] = pd.to_numeric(df[f'Montant{col}'], errors='coerce').fillna(0.0)
            else:
                df[f'Montant{col}'] = 0.0

        # Normalisation CompteNum
        if 'CompteNum' in df.columns:
            df['CompteNum'] = df['CompteNum'].astype(str).str.strip()
        else:
            df['CompteNum'] = ''

        # Date Analyse
        if 'EcritureDate' in df.columns:
            # Format FEC standard souvent YYYYMMDD
            df['Date_Analyse'] = pd.to_datetime(df['EcritureDate'], format='%Y%m%d', errors='coerce')
            df = df.dropna(subset=['Date_Analyse'])
        else:
            return None

        return df

    except Exception as e:
        st.error(f"Erreur lecture fichier {uploaded_file.name} : {e}")
        return None


def calculer_indicateurs_mensuels(df):
    """
    Logique EXACTE de 'courbes_KPI.py' pour le calcul des agrégats.
    """
    df = df.set_index('Date_Analyse').sort_index()

    # Fréquence Mensuelle (ME = Month End)
    groupe_mois = df.groupby(pd.Grouper(freq='ME'))
    resultats = []

    for mois, data in groupe_mois:
        if data.empty: continue

        # 1. CA (70...) - Crédit - Débit
        mask_ca = data['CompteNum'].str.startswith('70')
        ca = (data.loc[mask_ca, 'MontantCredit'] - data.loc[mask_ca, 'MontantDebit']).sum()

        # 2. EBITDA (Produits Expl - Charges Expl)
        mask_prod_expl = data['CompteNum'].str.match(r'^(70|71|72|73|74)')
        prod_expl = (data.loc[mask_prod_expl, 'MontantCredit'] - data.loc[mask_prod_expl, 'MontantDebit']).sum()

        mask_chg_expl = data['CompteNum'].str.match(r'^(60|61|62|63|64)')
        chg_expl = (data.loc[mask_chg_expl, 'MontantDebit'] - data.loc[mask_chg_expl, 'MontantCredit']).sum()

        ebitda = prod_expl - chg_expl

        # 3. Résultat Net (Classe 7 - Classe 6)
        mask_cl7 = data['CompteNum'].str.startswith('7')
        total_prod = (data.loc[mask_cl7, 'MontantCredit'] - data.loc[mask_cl7, 'MontantDebit']).sum()

        mask_cl6 = data['CompteNum'].str.startswith('6')
        total_chg = (data.loc[mask_cl6, 'MontantDebit'] - data.loc[mask_cl6, 'MontantCredit']).sum()

        resultat = total_prod - total_chg

        # 4. Flux Tréso (Classe 5)
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
        # Trésorerie cumulée (Position)
        df_res['Treso_Cumulee'] = df_res['Flux_Treso'].cumsum()

    return df_res


# --- 2. NOUVELLE LOGIQUE PRÉDICTIVE (MATHÉMATIQUE) ---

def generate_predictions(df_history, months_to_predict):
    """
    Génère des prédictions basées sur une Régression Linéaire (Tendance)
    + Ajout de la Moyenne Mensuelle (Saisonnalité simplifiée).
    """
    if df_history.empty or len(df_history) < 3:
        return None

    df = df_history.copy()

    # On crée un index numérique pour la régression (0, 1, 2...)
    df['Time_Index'] = np.arange(len(df))

    # Création des dates futures
    last_date = df.index[-1]
    future_dates = pd.date_range(start=last_date, periods=months_to_predict + 1, freq='ME')[1:]

    predictions = pd.DataFrame(index=future_dates)
    future_time_index = np.arange(len(df), len(df) + months_to_predict).reshape(-1, 1)

    indicators = ['CA', 'EBITDA', 'Resultat', 'Treso_Cumulee']

    for col in indicators:
        # 1. Calcul de la Tendance (Régression Linéaire)
        X = df[['Time_Index']]
        y = df[col].fillna(0)

        model = LinearRegression()
        model.fit(X, y)

        trend_future = model.predict(future_time_index)

        # 2. Ajout de bruit/variance (optionnel, pour réalisme) ou saisonnalité simple
        # Ici on reste sur la tendance pure pour être propre, ou on projette la moyenne des derniers mois
        # Pour faire simple et robuste : Tendance Linéaire
        predictions[col] = trend_future

    return predictions


# --- 3. ANALYSE IA (GEMINI - QUALITATIF SEULEMENT) ---

def get_gemini_analysis(api_key, context_text):
    if not api_key:
        return "Veuillez entrer une clé API pour obtenir l'analyse IA."

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-1.5-flash')

    prompt = f"""
    Tu es un expert financier. Analyse les données comptables suivantes (Historique + Prédiction mathématique).
    Donne ton avis sur :
    1. La tendance globale (Croissance ou Déclin ?)
    2. La santé de la trésorerie.
    3. Les risques potentiels visibles dans les chiffres.

    Données résumées :
    {context_text}

    Réponds de manière concise (5-6 phrases max) et professionnelle.
    """
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Erreur Gemini : {e}"


# --- 4. INTERFACE STREAMLIT ---

# Sidebar
st.sidebar.header("Paramètres")
api_key = st.sidebar.text_input("Clé API Gemini", type="password")
uploaded_files = st.sidebar.file_uploader("Fichiers FEC (.txt, .csv)", accept_multiple_files=True)
horizon_years = st.sidebar.slider("Horizon prédiction (années)", 1, 3, 1)

st.title("📊 Dashboard Financier : Historique & Prévisions")

if uploaded_files:
    all_dfs = []
    for file in uploaded_files:
        df_clean = nettoyer_et_preparer_donnees(file)
        if df_clean is not None:
            all_dfs.append(df_clean)

    if all_dfs:
        # 1. Fusionner tout l'historique brut
        df_global = pd.concat(all_dfs, ignore_index=True)

        # 2. Calculer les KPIs Mensuels (Votre Algo)
        df_history = calculer_indicateurs_mensuels(df_global)

        if not df_history.empty:
            # 3. Générer les Prédictions Mathématiques
            months_pred = horizon_years * 12
            df_pred = generate_predictions(df_history, months_pred)

            # --- AFFICHAGE ---

            # KPI Cards (Dernier mois connu)
            last_month = df_history.iloc[-1]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Dernier CA Mensuel", f"{last_month['CA']:,.0f} €")
            c2.metric("Dernier EBITDA", f"{last_month['EBITDA']:,.0f} €")
            c3.metric("Dernier Résultat", f"{last_month['Resultat']:,.0f} €")
            c4.metric("Trésorerie Actuelle", f"{last_month['Treso_Cumulee']:,.0f} €")

            st.markdown("---")

            # Graphiques Interactifs (Plotly)
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

                    # Historique (Ligne Pleine)
                    fig.add_trace(go.Scatter(
                        x=df_history.index,
                        y=df_history[config['col']],
                        mode='lines',
                        name='Historique',
                        line=dict(color=config['color'], width=3)
                    ))

                    # Prédiction (Ligne Pointillée)
                    if df_pred is not None:
                        # Pour connecter visuellement, on ajoute le dernier point historique
                        last_point = df_history.iloc[[-1]][[config['col']]]
                        pred_combined = pd.concat([last_point, df_pred[[config['col']]]])

                        fig.add_trace(go.Scatter(
                            x=pred_combined.index,
                            y=pred_combined[config['col']],
                            mode='lines',
                            name='Prédiction (Tendance)',
                            line=dict(color=config['color'], width=3, dash='dot')
                        ))

                    fig.update_layout(
                        title=config['title'],
                        xaxis_title="Date",
                        yaxis_title="Montant (€)",
                        height=350,
                        margin=dict(l=20, r=20, t=40, b=20)
                    )
                    st.plotly_chart(fig, use_container_width=True)

            # --- ANALYSE IA ---
            st.subheader("🤖 L'Analyse de l'Expert (IA)")
            if st.button("Lancer l'analyse textuelle avec Gemini"):
                with st.spinner("Analyse en cours..."):
                    # Préparation résumé pour IA
                    summary = f"""
                    Derniers 12 mois CA moyen: {df_history['CA'].tail(12).mean():.0f}
                    Dernière Trésorerie: {last_month['Treso_Cumulee']:.0f}
                    Tendance CA prévue (fin horizon): {df_pred['CA'].iloc[-1]:.0f}
                    """
                    analysis = get_gemini_analysis(api_key, summary)
                    st.info(analysis)

        else:
            st.warning("Données insuffisantes pour calculer les indicateurs.")