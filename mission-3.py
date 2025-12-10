import pandas as pd
import numpy as np
import os
import glob
import warnings
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans

warnings.filterwarnings('ignore')


class FinancialAuditorV9:
    def __init__(self, root_dir='database/fec_anonymes_csv/'):
        self.root_dir = root_dir
        self.companies_data = {}
        self.benchmarks = {}

        # Dictionnaire PCG Expert
        self.PCG = {
            'CAPITAUX_PROPRES': ['10', '11', '12'],
            'EMPRUNTS': ['16'],
            'IMMO_NET': ['2'],
            'STOCKS': ['3'],
            'CLIENTS': ['411', '413', '416', '418'],
            'FOURNISSEURS': ['401', '403', '408'],
            'FISCAL_SOCIAL_DETTE': ['42', '43', '44'],
            'DISPONIBILITES': ['512', '53'],
            'VENTES_MARCHANDISES': ['707'],
            'PRODUCTION_VENDUE': ['701', '702', '703', '704', '705', '706'],
            'ACHATS_MARCHANDISES': ['607'],
            'ACHATS_MATIERES': ['601', '602'],
            'VARIATION_STOCK': ['603'],
            'ACHATS_NON_STOCKES': ['606'],
            'SOUS_TRAITANCE': ['611'],
            'LOCATIONS': ['613', '614'],
            'ENTRETIEN': ['615'],
            'ASSURANCES': ['616'],
            'DIVERS_GESTION': ['618', '623', '624', '625', '626', '627', '628'],
            'PERSONNEL_EXTERIEUR': ['621'],
            'HONORAIRES': ['622'],
            'IMPOTS_TAXES': ['63'],
            'SALAIRES': ['641'],
            'CHARGES_SOCIALES': ['645'],
            'DOTATIONS_AMORT': ['681'],
            'CHARGES_FINANCIERES': ['66'],
            'PRODUITS_FINANCIERS': ['76'],
            'EXCEPTIONNEL': ['67', '77'],
            'IMPOT_SOCIETE': ['695']
        }

    # ==========================================================================
    # 1. INTELLIGENCE TEMPORELLE (Chargement & Découpage)
    # ==========================================================================

    def load_data(self):
        print("█ CHARGEMENT AVEC DÉCOUPAGE TEMPOREL (ANNÉE PAR ANNÉE)...")
        if not os.path.exists(self.root_dir):
            print(f"ERREUR: Dossier {self.root_dir} introuvable.")
            return

        for company_folder in sorted(os.listdir(self.root_dir)):
            path = os.path.join(self.root_dir, company_folder)
            if os.path.isdir(path):
                self._process_company_history(path, company_folder)

        print(f"✅ {len(self.companies_data)} entreprises analysées sur plusieurs exercices.")

    def _process_company_history(self, folder_path, company_id):
        all_files = glob.glob(os.path.join(folder_path, "*.csv"))
        df_list = []

        # 1. Lecture brute de tous les fichiers
        for f in all_files:
            try:
                df = pd.read_csv(f, dtype=str, low_memory=False, encoding='latin1')
                if len(df.columns) < 2:
                    df = pd.read_csv(f, dtype=str, low_memory=False, encoding='latin1', sep=';')

                df.columns = [c.replace(' ', '').replace('_', '').upper() for c in df.columns]

                # Nettoyage numérique
                for col in ['DEBIT', 'CREDIT']:
                    if col in df.columns:
                        df[col] = pd.to_numeric(
                            df[col].astype(str).str.replace(',', '.').str.replace(r'[^\d\.\-]', '', regex=True),
                            errors='coerce'
                        ).fillna(0)

                # Conversion Date pour découpage
                if 'ECRITUREDATE' in df.columns:
                    df['DATE'] = pd.to_datetime(df['ECRITUREDATE'], format='%Y%m%d', errors='coerce')
                    df['ANNEE'] = df['DATE'].dt.year

                df_list.append(df)
            except:
                continue

        if not df_list: return

        full_df = pd.concat(df_list, ignore_index=True)

        # 2. Découpage par Année Fiscale
        # On ne garde que les années avec assez de données (>100 lignes) pour éviter les reliquats
        years = full_df['ANNEE'].value_counts()
        valid_years = years[years > 100].index.sort_values(ascending=False)  # Du plus récent au plus vieux

        history = {}
        for year in valid_years:
            yearly_data = full_df[full_df['ANNEE'] == year]
            metrics = self._analyze_financials(yearly_data, company_id, year)
            history[year] = metrics

        if history:
            # On stocke l'historique complet pour cette entreprise
            # La clé principale pointe vers la DERNIÈRE année (N), mais contient N-1
            last_year = valid_years[0]
            current = history[last_year]

            if len(valid_years) > 1:
                prev_year = valid_years[1]
                current['PREVIOUS'] = history[prev_year]
            else:
                current['PREVIOUS'] = None

            self.companies_data[company_id] = current

    # ==========================================================================
    # 2. CALCULATEUR FINANCIER (SIG & FLUX)
    # ==========================================================================

    def _analyze_financials(self, df, company_id, year):
        f = {'ID': company_id, 'YEAR': year}

        def sum_acc(roots, mode='solde'):  # mode: solde (C-D ou D-C selon sens), debit, credit
            total = 0
            for root in roots:
                mask = df['COMPTENUM'].str.startswith(root, na=False)
                d = df.loc[mask, 'DEBIT'].sum()
                c = df.loc[mask, 'CREDIT'].sum()

                if mode == 'produit':
                    total += (c - d)
                elif mode == 'charge':
                    total += (d - c)
                elif mode == 'actif':
                    total += (d - c)
                elif mode == 'passif':
                    total += (c - d)
                elif mode == 'solde_compte':
                    total += (c - d)
            return total

        # --- P&L ---
        f['CA'] = sum_acc(self.PCG['VENTES_MARCHANDISES'] + self.PCG['PRODUCTION_VENDUE'], 'produit')
        f['Achats_Conso'] = sum_acc(self.PCG['ACHATS_MARCHANDISES'] + self.PCG['ACHATS_MATIERES'], 'charge') - sum_acc(self.PCG['VARIATION_STOCK'], 'produit')
        f['Marge_Brute'] = f['CA'] - f['Achats_Conso']

        f['Charges_Ext'] = sum_acc(self.PCG['ACHATS_NON_STOCKES'] + self.PCG['SOUS_TRAITANCE'] + self.PCG['LOCATIONS'] +
                                   self.PCG['ENTRETIEN'] + self.PCG['ASSURANCES'] + self.PCG['DIVERS_GESTION'] +
                                   self.PCG['PERSONNEL_EXTERIEUR'] + self.PCG['HONORAIRES'], 'charge')

        f['Masse_Sal'] = sum_acc(self.PCG['SALAIRES'] + self.PCG['CHARGES_SOCIALES'], 'charge')
        f['Impots'] = sum_acc(self.PCG['IMPOTS_TAXES'], 'charge')

        f['EBE'] = f['Marge_Brute'] - f['Charges_Ext'] - f['Masse_Sal'] - f['Impots']

        f['Dot_Amort'] = sum_acc(self.PCG['DOTATIONS_AMORT'], 'charge')
        f['Frais_Fin'] = sum_acc(self.PCG['CHARGES_FINANCIERES'], 'charge')
        f['Resultat_Net'] = f['EBE'] - f['Dot_Amort'] - f['Frais_Fin']  # Simplifié

        # --- BILAN ---
        f['Tréso_Active'] = sum_acc(self.PCG['DISPONIBILITES'], 'actif')

        # Calcul dette bancaire précise (Solde créditeur 16 + Solde créditeur 512)
        f['Dette_LT'] = sum_acc(self.PCG['EMPRUNTS'], 'passif')

        # Check découverts
        decouvert = 0
        for acc in ['512']:
            mask = df['COMPTENUM'].str.startswith(acc, na=False)
            solde = df.loc[mask, 'CREDIT'].sum() - df.loc[mask, 'DEBIT'].sum()
            if solde > 0: decouvert += solde

        f['Dette_Nette'] = f['Dette_LT'] + decouvert - f['Tréso_Active']
        f['Tresorerie_Nette'] = f['Tréso_Active'] - decouvert
        f['Capitaux_Propres'] = sum_acc(self.PCG['CAPITAUX_PROPRES'], 'passif') + f['Resultat_Net']

        # --- RATIOS GRANULAIRES ---
        if f['CA'] > 0:
            f['R_Matiere'] = f['Achats_Conso'] / f['CA'] * 100
            f['R_Masse_Sal'] = f['Masse_Sal'] / f['CA'] * 100
            f['R_Energie'] = sum_acc(['6061'], 'charge') / f['CA'] * 100
            f['R_Loyer'] = sum_acc(['613'], 'charge') / f['CA'] * 100
            f['R_Interim'] = sum_acc(['621'], 'charge') / f['CA'] * 100
        else:
            f['R_Matiere'] = 0
            f['R_Masse_Sal'] = 0
            f['R_Energie'] = 0
            f['R_Loyer'] = 0
            f['R_Interim'] = 0

        f['Prime_Cost'] = f['R_Matiere'] + f['R_Masse_Sal'] + f['R_Interim']

        return f

    # ==========================================================================
    # 3. BENCHMARKING & ANALYSE DYNAMIQUE
    # ==========================================================================

    def run_benchmarks(self):
        print("█ DÉFINITION DES CLUSTERS ET CALCUL DES TENDANCES...")
        # On ne prend que les données de l'année N pour le clustering
        data_n = list(self.companies_data.values())
        if not data_n: return pd.DataFrame()

        df = pd.DataFrame(data_n)
        df.set_index('ID', inplace=True)

        # Clustering
        X = df[['CA', 'Prime_Cost', 'EBE']].fillna(0)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        kmeans = KMeans(n_clusters=3, random_state=42)
        df['CLUSTER'] = kmeans.fit_predict(X_scaled)

        # Calcul des cibles par cluster (Médianes performantes)
        for c in range(3):
            sub = df[df['CLUSTER'] == c]
            self.benchmarks[c] = {
                'CA_MOYEN': sub['CA'].mean(),
                'MATIERE_CIBLE': sub['R_Matiere'].quantile(0.25),
                'RH_CIBLE': sub['R_Masse_Sal'].quantile(0.25),
                'ENERGIE_CIBLE': sub['R_Energie'].median(),
                'LOYER_CIBLE': sub['R_Loyer'].median()
            }

        return df

    def get_trend_icon(self, current, previous, metric, inverse=False):
        """Retourne une flèche de tendance N vs N-1"""
        if previous is None or previous[metric] == 0: return "  "

        delta = (current[metric] - previous[metric]) / abs(previous[metric]) * 100

        if abs(delta) < 2: return "➡️ stable"

        is_good = (delta > 0) if not inverse else (delta < 0)
        icon = "↗️" if delta > 0 else "↘️"

        if metric in ['CA', 'EBE', 'Resultat_Net']:  # Plus c'est haut mieux c'est
            color = ""
        else:  # Pour les charges, une hausse est mauvaise
            pass

        return f"{icon} {delta:+.1f}%"

    # ==========================================================================
    # 4. GÉNÉRATION DU RAPPORT
    # ==========================================================================

    def generate_full_report(self):
        self.load_data()
        df = self.run_benchmarks()

        if df.empty: return

        print("\n" + "=" * 100)
        print(f"STC OMNISCIENT V9.0 - AUDIT STRATÉGIQUE & TEMPOREL ({len(df)} Dossiers)")
        print("=" * 100 + "\n")

        for cid in df.index:
            curr = self.companies_data[cid]
            prev = curr.get('PREVIOUS')
            stats = self.benchmarks[df.loc[cid, 'CLUSTER']]

            # --- HEADER ---
            print(f"DOSSIER : {cid}  (Exercice {curr['YEAR']})")
            trend_ca = self.get_trend_icon(curr, prev, 'CA')
            print(f"ACTIVITÉ: {curr['CA']:,.0f} € {trend_ca}  |  CLUSTER: Type {df.loc[cid, 'CLUSTER']}")

            # --- SCORE & DIAGNOSTIC ---
            score = 50
            if curr['EBE'] > 0: score += 20
            if curr['Tresorerie_Nette'] > 0: score += 10
            if prev and curr['EBE'] > prev['EBE']: score += 10

            status = "ROBUSTE" if score >= 70 else "FRAGILE" if score >= 50 else "CRITIQUE"
            symbol = "🟢" if score >= 70 else "🟠" if score >= 50 else "🔴"

            print(f"SANTÉ   : {symbol} {status} (Score: {score}/100)")
            print("-" * 100)

            # --- 1. ANALYSE DYNAMIQUE DU RÉSULTAT ---
            print(" [1] DYNAMIQUE DE RENTABILITÉ")
            print(f"     Marge Brute : {curr['Marge_Brute']:,.0f} € ({curr['Marge_Brute'] / curr['CA'] * 100:.1f}%) {self.get_trend_icon(curr, prev, 'Marge_Brute')}")
            print(f"     E.B.E.      : {curr['EBE']:,.0f} € ({curr['EBE'] / curr['CA'] * 100:.1f}%) {self.get_trend_icon(curr, prev, 'EBE')}")
            print(f"     Résultat Net: {curr['Resultat_Net']:,.0f} € {self.get_trend_icon(curr, prev, 'Resultat_Net')}")

            if prev:
                delta_ebe = curr['EBE'] - prev['EBE']
                if delta_ebe < -10000:
                    print(f"     ⚠️ ALERTE TENDANCE: L'EBE s'est dégradé de {delta_ebe:,.0f} € en un an.")

            print("")

            # --- 2. FOCUS COÛTS & DÉRAPAGES ---
            print(" [2] STRUCTURE DE COÛTS & DÉRAPAGES")

            def check_ratio(label, key, target):
                val = curr[key]
                trend = self.get_trend_icon(curr, prev, key, inverse=True)
                gap = val - target
                state = "✅" if gap < 2 else "⚠️ DÉRAPAGE" if gap > 5 else "⚠️ A SURVEILLER"
                print(f"     {label:<12} : {val:.1f}% (Cible {target:.1f}%) {state}  Trend: {trend}")
                return gap

            gap_mat = check_ratio("Matières", 'R_Matiere', stats['MATIERE_CIBLE'])
            gap_rh = check_ratio("Salaires", 'R_Masse_Sal', stats['RH_CIBLE'])
            gap_nrg = check_ratio("Énergie", 'R_Energie', stats['ENERGIE_CIBLE'])

            print("")

            # --- 3. BILAN & SOLVABILITÉ ---
            print(" [3] BILAN & TENSION DE TRÉSORERIE")
            print(f"     Cash Dispo  : {curr['Tresorerie_Nette']:,.0f} €")
            print(f"     Dette Nette : {curr['Dette_Nette']:,.0f} €")

            if curr['Tresorerie_Nette'] < 0:
                print("     >>> TENSION: Découvert bancaire utilisé.")

            print("")

            # --- 4. PLAN D'ACTION INTELLIGENT ---
            print(" [4] RECOMMANDATIONS PRIORITAIRES")
            actions = []

            # Recommandations basées sur les ÉCARTS et les TENDANCES
            if curr['EBE'] < 0:
                manque = abs(curr['EBE'])
                actions.append(f"[URGENT] STOPPER LES PERTES : Votre modèle perd {manque:,.0f} €/an avant même de payer la dette. Audit prix/volume urgent.")

            if gap_mat > 5:
                gain = curr['CA'] * (gap_mat / 100)
                actions.append(f"[MARGE] Casser les coûts matière : Vous êtes {gap_mat:.1f} pts au dessus du marché. Gain potentiel: {gain:,.0f} €.")

            if gap_rh > 5:
                gain = curr['CA'] * (gap_rh / 100)
                actions.append(f"[RH] Productivité faible : La masse salariale absorbe trop de valeur. Gain potentiel: {gain:,.0f} €.")

            if curr['R_Energie'] > stats['ENERGIE_CIBLE'] * 1.5:
                actions.append(f"[COUTS] Alerte Énergie : Vos coûts sont {curr['R_Energie'] / stats['ENERGIE_CIBLE']:.1f}x supérieurs à la norme.")

            if not actions:
                print("     >> Félicitations. Gestion saine et alignée sur les standards.")
            else:
                for a in actions:
                    print(f"     👉 {a}")

            print("\n" + "." * 100 + "\n")


if __name__ == "__main__":
    auditor = FinancialAuditorV9()
    auditor.generate_full_report()