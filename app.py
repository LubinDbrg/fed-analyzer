import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

# Configuration de la page
st.set_page_config(page_title="Dashboard Financier", layout="wide")

def get_financial_data():
    """
    Génère les données avec une logique financière corrigée :
    - Le CA baisse puis se stabilise.
    - L'EBITDA est corrélé au CA (marge variable).
    - La Trésorerie est CUMULATIVE (elle baisse quand le résultat est négatif).
    """
    dates = pd.date_range(start='2017-01-01', periods=10, freq='AS') # AS = Début d'année

    # 1. Chiffre d'Affaires (CA)
    # Scénario : Forte baisse, crise, puis stabilisation
    ca_data = [850000, 680000, 450000, 450000, 350000, 470000, 260000, 280000, 250000, 245000]

    # 2. EBITDA (Rentabilité d'Exploitation)
    # On simule une marge qui s'effondre puis remonte grâce à une restructuration
    marges = [0.03, -0.01, -0.02, -0.03, 0.00, 0.12, 0.05, -0.04, 0.08, 0.18]
    ebitda_data = [ca * m for ca, m in zip(ca_data, marges)]

    # 3. Résultat Net
    # EBITDA - Charges fixes estimées (ex: 10k par an)
    charges_fixes = 10000
    net_result_data = [e - charges_fixes for e in ebitda_data]

    # 4. Trésorerie (CORRECTION MAJEURE ICI)
    # La trésorerie est un stock qui varie selon le flux du Résultat Net
    cash_initial = 150000
    cash_data = []
    current_cash = cash_initial
    
    for resultat in net_result_data:
        current_cash += resultat  # Si résultat négatif, le cash diminue
        cash_data.append(current_cash)

    df = pd.DataFrame({
        'Date': dates,
        'CA': ca_data,
        'EBITDA': ebitda_data,
        'Net_Result': net_result_data,
        'Cash': cash_data
    })
    return df

def plot_dashboard(df):
    """
    Crée les graphiques Matplotlib avec le style exact demandé.
    """
    # Création de la figure globale
    fig, axs = plt.subplots(2, 2, figsize=(16, 10))
    fig.patch.set_facecolor('white') # Fond blanc
    
    # Titre global géré par Streamlit, mais on peut le mettre sur le plot si besoin
    # fig.suptitle('Tableau de Bord Financier : 018403', fontsize=16, fontweight='bold')

    # --- Fonction de style interne ---
    def style_ax(ax, title):
        ax.set_title(title, fontweight='bold', fontsize=12, pad=10)
        ax.grid(True, linestyle='--', alpha=0.6, color='gray')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%Y'))
        plt.setp(ax.get_xticklabels(), rotation=30, ha='right')
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        ax.spines['bottom'].set_color('black')
        ax.spines['left'].set_color('black')

    # --- 1. CA (Barres Bleues) ---
    axs[0, 0].bar(df['Date'], df['CA'], width=200, color='#1f77b4', alpha=0.9)
    style_ax(axs[0, 0], "Chiffre d'Affaires (CA)")

    # --- 2. EBITDA (Ligne Orange) ---
    axs[0, 1].plot(df['Date'], df['EBITDA'], color='#ff7f0e', marker='o', linewidth=2.5, markersize=6)
    axs[0, 1].axhline(0, color='black', linewidth=1)
    style_ax(axs[0, 1], "EBITDA (Rentabilité d'Exploitation)")

    # --- 3. Résultat Net (Barres Vertes/Rouges) ---
    colors = ['#2ca02c' if x >= 0 else '#d62728' for x in df['Net_Result']]
    axs[1, 0].bar(df['Date'], df['Net_Result'], width=200, color=colors, alpha=0.9)
    axs[1, 0].axhline(0, color='black', linewidth=1)
    style_ax(axs[1, 0], "Résultat Net")

    # --- 4. Trésorerie (Aire Violette) ---
    axs[1, 1].plot(df['Date'], df['Cash'], color='#9467bd', linewidth=2)
    axs[1, 1].fill_between(df['Date'], df['Cash'], color='#9467bd', alpha=0.3)
    # Ligne rouge pointillée pour marquer le niveau zéro (danger)
    axs[1, 1].axhline(0, color='red', linestyle=':', linewidth=1, alpha=0.7)
    style_ax(axs[1, 1], "Position de Trésorerie (Cumul)")

    plt.tight_layout()
    return fig

# --- Main App ---

st.title("Tableau de Bord Financier : 018403")
st.markdown("---")

# Récupération des données
df = get_financial_data()

# Affichage des indicateurs clés (KPIs) en haut
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("CA (Dernier)", f"{df['CA'].iloc[-1]:,.0f} €", delta=f"{df['CA'].iloc[-1] - df['CA'].iloc[-2]:,.0f}")
kpi2.metric("EBITDA (Dernier)", f"{df['EBITDA'].iloc[-1]:,.0f} €")
kpi3.metric("Résultat Net (Dernier)", f"{df['Net_Result'].iloc[-1]:,.0f} €")
kpi4.metric("Trésorerie Actuelle", f"{df['Cash'].iloc[-1]:,.0f} €", delta_color="normal")

st.markdown("---")

# Affichage du graphique Matplotlib
fig = plot_dashboard(df)
st.pyplot(fig)

# Affichage des données brutes (optionnel)
with st.expander("Voir les données sources"):
    st.dataframe(df.style.format("{:,.0f}"))
