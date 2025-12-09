import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import google.generativeai as genai
import json
import io

# Configuration de la page Streamlit
st.set_page_config(page_title="FEC Analyzer & Predictor", layout="wide")


# -----------------------------------------------------------------------------
# 1. FONCTIONS DE TRAITEMENT DES DONNÉES (BACKEND)
# -----------------------------------------------------------------------------

def load_and_merge_fec(uploaded_files):
    """
    Lit plusieurs fichiers FEC, normalise les colonnes et les fusionne.
    Gère les séparateurs '|' ou ';' et le format des nombres français (1 000,00).
    """
    all_data = []

    # Colonnes standards attendues dans un FEC (simplifié pour la démo)
    # En réalité, les noms peuvent varier (EcritureDate vs DateEcriture, etc.)
    required_cols = ['EcritureDate', 'CompteNum', 'Debit', 'Credit']

    for file in uploaded_files:
        try:
            # Les fichiers FEC sont souvent encodés en latin-1 ou cp1252 en France
            # On tente de lire avec séparateur '|' ou ';'
            content = file.getvalue().decode('latin-1')

            # Détection simple du séparateur
            sep = '|' if content.count('|') > content.count(';') else ';'

            df = pd.read_csv(
                io.StringIO(content),
                sep=sep,
                decimal=',',  # Important pour le format français "120,50"
                dtype={'CompteNum': str}  # Garder les comptes en string (ex: 401000)
            )

            # Normalisation basique des colonnes (nettoyage des espaces)
            df.columns = df.columns.str.strip()

            # Vérification des colonnes critiques
            missing_cols = [c for c in required_cols if c not in df.columns]
            if missing_cols:
                st.warning(f"Fichier {file.name} ignoré : colonnes manquantes {missing_cols}")
                continue

            # Conversion de la date
            # Format standard FEC : Souvent YYYYMMDD ou DD/MM/YYYY
            # On force la conversion
            df['EcritureDate'] = pd.to_datetime(df['EcritureDate'], errors='coerce')

            # Remplacer les NaN par 0 pour les calculs
            df['Debit'] = df['Debit'].fillna(0)
            df['Credit'] = df['Credit'].fillna(0)

            all_data.append(df)

        except Exception as e:
            st.error(f"Erreur lors de la lecture de {file.name}: {e}")

    if not all_data:
        return pd.DataFrame()

    return pd.concat(all_data, ignore_index=True)


def calculate_kpis(df):
    """
    Calcule les 4 KPIs annuels basés sur les règles du PCG.
    """
    if df.empty:
        return pd.DataFrame()

    # Extraction de l'année
    df['Year'] = df['EcritureDate'].dt.year

    # Nettoyage numéros de compte (garder que les chiffres au début)
    df['CompteClass'] = df['CompteNum'].astype(str).str[:3]  # ex: '707'
    df['RootClass'] = df['CompteNum'].astype(str).str[:1]  # ex: '7'

    # Groupement par année pour l'agrégation
    years = df['Year'].unique()
    years.sort()

    kpi_data = []

    for year in years:
        mask_year = df['Year'] == year
        df_y = df[mask_year]

        # 1. Chiffre d'Affaires (CA) : Somme Crédit des comptes commençant par '70'
        # Note: On soustrait le débit (retours/avoirs) pour être précis
        ca_mask = df_y['CompteNum'].str.startswith('70')
        ca = df_y.loc[ca_mask, 'Credit'].sum() - df_y.loc[ca_mask, 'Debit'].sum()

        # 2. EBITDA (Approximation simplifiée)
        # Produits d'exploitation (Cl. 7) - Charges d'exploitation (Cl. 6)
        # + Dotations Amortissements (Cl. 68) -> On les rajoute car elles ont été soustraites dans les charges
        prod_expl = df_y.loc[df_y['RootClass'] == '7', 'Credit'].sum() - df_y.loc[
            df_y['RootClass'] == '7', 'Debit'].sum()
        charge_expl = df_y.loc[df_y['RootClass'] == '6', 'Debit'].sum() - df_y.loc[
            df_y['RootClass'] == '6', 'Credit'].sum()

        # Amortissements (Compte 68)
        amort = df_y.loc[df_y['CompteNum'].str.startswith('68'), 'Debit'].sum() - df_y.loc[
            df_y['CompteNum'].str.startswith('68'), 'Credit'].sum()

        ebitda = (prod_expl - charge_expl) + amort

        # 3. Résultat Net
        # Total Produits (Cl. 7) - Total Charges (Cl. 6)
        # Pour faire simple on prend la différence globale.
        # Une méthode plus précise serait le solde de la classe 12.
        resultat_net = prod_expl - charge_expl

        # 4. Trésorerie
        # Solde des comptes de classe 5 (Banque, Caisse)
        # Actif (Debit) - Passif (Credit)
        treso_mask = df_y['RootClass'] == '5'
        tresorerie = df_y.loc[treso_mask, 'Debit'].sum() - df_y.loc[treso_mask, 'Credit'].sum()

        kpi_data.append({
            'Année': int(year),
            'CA': round(ca, 2),
            'EBITDA': round(ebitda, 2),
            'Resultat_Net': round(resultat_net, 2),
            'Tresorerie': round(tresorerie, 2)
        })

    return pd.DataFrame(kpi_data)


# -----------------------------------------------------------------------------
# 2. INTÉGRATION IA (GEMINI)
# -----------------------------------------------------------------------------

def get_gemini_predictions(api_key, sector, history_df, horizon):
    """
    Envoie l'historique financier à Gemini et demande une projection JSON.
    """
    if not api_key:
        st.error("Clé API manquante.")
        return None

    genai.configure(api_key=api_key)

    # Conversion de l'historique en CSV string pour le prompt
    history_csv = history_df.to_csv(index=False)

    # Prompt Engineering
    prompt = f"""
    Tu es un analyste financier expert. Voici les données financières historiques d'une entreprise du secteur : {sector}.

    Données historiques (CSV) :
    {history_csv}

    TACHE :
    Prédire les valeurs financières pour les {horizon} prochaines années (Année N+1 à N+{horizon}) pour les indicateurs suivants :
    - CA
    - EBITDA
    - Resultat_Net
    - Tresorerie

    Considère les tendances du secteur '{sector}' (inflation, croissance typique, risques) pour ajuster tes prédictions de manière réaliste.

    FORMAT DE SORTIE OBLIGATOIRE :
    Tu dois répondre UNIQUEMENT avec un objet JSON valide. Pas de texte avant ni après, pas de balises markdown (```json).
    Le format doit être une liste d'objets :
    [
        {{"Année": 2025, "CA": 100000, "EBITDA": 20000, "Resultat_Net": 15000, "Tresorerie": 5000}},
        ...
    ]
    """

    model = genai.GenerativeModel('gemini-1.5-flash')  # Modèle rapide et efficace pour les chiffres

    try:
        response = model.generate_content(prompt)
        text_response = response.text.strip()

        # Nettoyage si Gemini ajoute des backticks malgré l'instruction
        if text_response.startswith("```json"):
            text_response = text_response.replace("```json", "").replace("```", "")

        predictions = json.loads(text_response)
        return pd.DataFrame(predictions)

    except Exception as e:
        st.error(f"Erreur API Gemini : {e}")
        return None


# -----------------------------------------------------------------------------
# 3. INTERFACE UTILISATEUR (STREAMLIT)
# -----------------------------------------------------------------------------

# --- Sidebar ---
st.sidebar.title("⚙️ Paramètres")
api_key = st.sidebar.text_input("Clé API Gemini", type="password", help="Obtenez-la sur Google AI Studio")

st.sidebar.header("📁 Données FEC")
uploaded_files = st.sidebar.file_uploader(
    "Charger fichiers FEC (.txt, .csv)",
    accept_multiple_files=True,
    type=['csv', 'txt']
)

st.sidebar.header("📊 Contexte Entreprise")
sector = st.sidebar.selectbox("Secteur d'activité",
                              ["Retail / Commerce", "Technologie / SaaS", "Industrie / BTP", "Services",
                               "Restauration/Hôtellerie"]
                              )
horizon = st.sidebar.slider("Horizon de prédiction (années)", 1, 3, 2)

run_analysis = st.sidebar.button("🚀 Lancer l'analyse")

# --- Main Area ---
st.title("📈 Analyse Financière Prédictive (FEC + IA)")
st.markdown("""
Cette application transforme vos fichiers d'écritures comptables (FEC) en insights financiers.
Elle calcule l'historique et utilise **Google Gemini** pour projeter vos résultats futurs.
""")

if run_analysis and uploaded_files and api_key:
    with st.spinner("Traitement des fichiers comptables en cours..."):
        # 1. Parsing et Fusion
        df_merged = load_and_merge_fec(uploaded_files)

        if not df_merged.empty:
            # 2. Calcul Historique
            df_history = calculate_kpis(df_merged)

            st.subheader("📝 Historique Financier (Calculé)")
            st.dataframe(df_history.style.format("{:,.2f} €"), use_container_width=True)

            # 3. Prédiction IA
            with st.spinner("L'IA analyse les tendances et génère les prédictions..."):
                df_pred = get_gemini_predictions(api_key, sector, df_history, horizon)

            if df_pred is not None:
                st.success("Analyse terminée avec succès !")

                # Fusionner pour l'affichage (ajout d'une colonne type)
                df_history['Type'] = 'Historique'
                df_pred['Type'] = 'Prédiction'

                # S'assurer que les années s'enchaînent correctement pour le graphique
                # On concatène tout
                full_df = pd.concat([df_history, df_pred], ignore_index=True)

                st.subheader("🔮 Projections Financières")

                # 4. Visualisation (4 Graphiques)
                indicators = ['CA', 'EBITDA', 'Resultat_Net', 'Tresorerie']
                cols = st.columns(2)  # Grille 2x2

                for i, kpi in enumerate(indicators):
                    col = cols[i % 2]
                    with col:
                        fig = go.Figure()

                        # Ligne Historique
                        hist_data = full_df[full_df['Type'] == 'Historique']
                        fig.add_trace(go.Scatter(
                            x=hist_data['Année'],
                            y=hist_data[kpi],
                            mode='lines+markers',
                            name='Historique',
                            line=dict(color='blue', width=3)
                        ))

                        # Ligne Prédiction
                        # Pour lier visuellement, on prend le dernier point historique + les prédictions
                        last_hist = hist_data.iloc[[-1]]
                        pred_data = full_df[full_df['Type'] == 'Prédiction']
                        combined_pred = pd.concat([last_hist, pred_data])

                        fig.add_trace(go.Scatter(
                            x=combined_pred['Année'],
                            y=combined_pred[kpi],
                            mode='lines+markers',
                            name='Prédiction IA',
                            line=dict(color='orange', width=3, dash='dot')
                        ))

                        fig.update_layout(
                            title=f"Évolution {kpi}",
                            xaxis_title="Année",
                            yaxis_title="Montant (€)",
                            template="plotly_white",
                            height=350
                        )
                        st.plotly_chart(fig, use_container_width=True)

                # Explication textuelle (facultatif mais utile)
                st.markdown("---")
                st.caption(
                    f"*Les prédictions sont générées par Gemini 1.5 Flash basées sur le secteur '{sector}' et l'historique comptable fourni.*")

        else:
            st.error("Impossible d'extraire des données valides des fichiers fournis.")

elif run_analysis and not api_key:
    st.warning("Veuillez entrer votre clé API Gemini dans la barre latérale.")
elif run_analysis and not uploaded_files:
    st.warning("Veuillez charger au moins un fichier FEC.")