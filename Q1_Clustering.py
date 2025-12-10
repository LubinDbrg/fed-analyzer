# Créé par cleme, le 08/12/2025 en Python 3.7
import csv
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import pickle
import os
import sys


def CA(file):
    List = file[["CompteNum", "Debit", "Credit"]].values.tolist()
    CA_Debit = 0
    CA_Credit = 0
    for i in List:
        if str(i[0])[:2] == "70":
            i[1] = str(i[1]).replace ( "," , "." )
            i[2] = str(i[2]).replace ( "," , "." )
            CA_Debit += float(i[1])
            CA_Credit += float(i[2])
    CA = CA_Credit - CA_Debit
    return (CA)


def ratio_endettement(file):
    List = file[["CompteNum", "Debit", "Credit"]].values.tolist()
    capital = 0
    dettes = 0
    for i in List:
        if str(i[0])[:2] in ["10","11","12","13","15"]:
            i[2] = str(i[2]).replace ( "," , "." )
            capital += float(i[2])
        elif str(i[0])[:2] in ["16"] or str(i[0])[:3] =="512" :
            i[2] = str(i[2]).replace ( "," , "." )
            dettes += float(i[2])
        elif str(i[0])[:2] in ["50","53"] or str(i[0])[:3] =="512" :
            i[1] = str(i[1]).replace ( "," , "." )
            dettes -= float(i[1])
    if capital != 0 :
        ratio_endettement = abs(dettes/capital)
    else:
        ratio_endettement = None
    return (ratio_endettement)


def Poids_Liquidité(file):
    List = file[["CompteNum", "Debit", "Credit"]].values.tolist()
    Tresorerie = 0
    Actif = 0
    for i in List:
        if str(i[0])[:3] == "512" or str(i[0])[:2] in ["50","53"] :
            i[1] = str(i[1]).replace ( "," , "." )
            Tresorerie += float(i[1])
        elif str(i[0])[:1] in ["1","2","3","4","5"] :
            i[1] = str(i[1]).replace ( "," , "." )
            Actif += float(i[1])
    if Actif != 0 :
        Poids_Liquidité = Tresorerie/Actif
    else:
        Poids_Liquidité = None
    return (Poids_Liquidité)


def Decouvert(file):
    List = file[["CompteNum", "Debit", "Credit"]].values.tolist()
    Solde = 0
    CA = 0
    for i in List:
        if str(i[0])[:3] == "512" :
            i[2] = str(i[2]).replace ( "," , "." )
            Solde += float(i[2])
        elif str(i[0])[:2] == "70" :
            i[2] = str(i[2]).replace ( "," , "." )
            CA += float(i[2])
    if CA != 0 :
        Decouvert = Solde/CA
    else:
        Decouvert = None
    return (Decouvert)


def Poids_provisions(file):
    List = file[["CompteNum", "Debit", "Credit"]].values.tolist()
    Solde = 0
    passif = 0
    for i in List:
        if str(i[0])[:2] == "15" :
            i[2] = str(i[2]).replace ( "," , "." )
            Solde += float(i[2])
        elif str(i[0])[:2] in ["10","11","12","13","14", "15", "16", "40", "43", "44", "45"] or str(i[0])[:3] in ["512", "419"] :
            i[2] = str(i[2]).replace ( "," , "." )
            passif += float(i[2])
    if passif != 0 :
        Poids_provisions = Solde/passif
    else:
        Poids_provisions = None
    return (Poids_provisions)


def Rotation_stock(file):
    List = file[["CompteNum", "Debit", "Credit"]].values.tolist()
    Solde = 0
    ventes = 0
    for i in List:
        if str(i[0])[:1] == "3" :
            i[1] = str(i[1]).replace ( "," , "." )
            Solde += float(i[1])
        elif str(i[0])[:2] == "60" :
            i[1] = str(i[1]).replace ( "," , "." )
            ventes += float(i[1])
    if ventes != 0:
        Rotation_stock = Solde/ventes
    else:
        Rotation_stock = None
    return (Rotation_stock)


def sommecsv(dossier_racine):
    all_data = []
    for filename in os.listdir(dossier_racine):
        if filename.lower().endswith('.csv'):
            file_path = os.path.join(dossier_racine, filename)

            try:
                df = pd.read_csv(file_path, sep=',', encoding='utf-8')
                all_data.append(df)

            except Exception as e:
                print(f"  Erreur lors du chargement de {file_path}: {e}")

        if all_data:
            df_final = pd.concat(all_data, ignore_index=True)
    return df_final



def csv_clustering(root_folder, output_file='donnees.csv'):

    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        headers = ["entreprise", "ratio d'endettement", "Poids liquidités",
                   "Poids provisions", "ratio découvert", "rotation des stock"]
        writer.writerow(headers)

        for dirpath, dirname, filename in os.walk(root_folder):
            try :
                file_df = sommecsv(dirpath)
                data = [
                    str(os.path.basename(dirpath)),
                    ratio_endettement(file_df),
                    Poids_Liquidité(file_df),
                    Poids_provisions(file_df),
                    Decouvert(file_df),
                    Rotation_stock(file_df)
                    ]
                writer.writerow(data)

            except Exception as e:
                    # Gestion des erreurs de lecture ou de calcul
                print(f"Erreur de traitement du fichier {filename}: {e}")

csv_clustering("fec_anonymes_csv")


# --- 1. CONFIGURATION ET CHEMINS DE SORTIE ---
FEATURES = ["ratio d'endettement", "Poids liquidités", "Poids provisions",
            "ratio découvert", "rotation des stock"]
K = 6
DB_PATH = 'donnees.csv'

# FICHIERS DE SORTIE DU MODÈLE
CENTRES_PATH = 'model_assets/model_centres.csv'
SCALER_PATH = 'model_assets/model_scaler.pkl'
PROFILES_RESULTS_PATH = 'model_assets/resultats_profilage_par_entreprise.csv'
FECS_PROFIL_INDIVIDUEL_PATH = 'model_assets/fecs_profils_individuels.csv' # NOUVEAU FICHIER

ASSETS_DIR = 'model_assets'
if not os.path.exists(ASSETS_DIR):
    os.makedirs(ASSETS_DIR, exist_ok=True)

# --- 2. FONCTIONS DE CLASSIFICATION (Identiques) ---

def classify_profiles_unique(cluster_centers_df):
    """ Classe les centres de clusters en profils uniques (logique fixe pour K=6). """
    K_local = len(cluster_centers_df)
    df_work = cluster_centers_df.copy()
    df_work['Original_Cluster'] = df_work.index
    df_work['Profil_Qualitatif'] = None
    df_work['Assigned'] = False

    profiles_priority = [
        ('ratio d\'endettement', 'Max', "1. L'Opportuniste (Max Dette)"),
        ('ratio d\'endettement', 'Min', "2. Le Conservateur (Min Dette)"),
        ('Poids provisions', 'Max', "3. Le Prévoyant (Max Anticipation)"),
        ('ratio découvert', 'Max', "4. Le Fragile (Découvert Critique)"),
        ('rotation des stock', 'Min', "5. Le Rigoureux (Max Efficacité)"),
        ('Poids liquidités', 'Min', "6. Le Tendu (Min Liquidité)"),
    ]

    def find_and_assign(df, criterion_col, extreme_type, name):
        df_temp = df[~df['Assigned']]
        if df_temp.empty: return df
        if extreme_type == 'Max': idx = df_temp[criterion_col].idxmax()
        else: idx = df_temp[criterion_col].idxmin()
        df_work.loc[idx, 'Profil_Qualitatif'] = name
        df_work.loc[idx, 'Assigned'] = True
        return df_work

    for i, (col, extreme, name) in enumerate(profiles_priority):
        if i < K_local:
            df_work = find_and_assign(df_work, col, extreme, name)

    return df_work[['Profil_Qualitatif', 'Original_Cluster']]

# --- 3. FONCTION PRINCIPALE : ENTRAÎNEMENT ET SAUVEGARDE DU MODÈLE ---

def train_model_and_save_assets(file_path=DB_PATH, K=K):
    """ Exécute le clustering et sauvegarde tous les assets, y compris les profils FEC individuels. """

    print(f"\n======== DÉBUT DE L'ENTRAÎNEMENT DU MODÈLE (K={K}) ========")

    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"ERREUR : Le fichier de base de données '{file_path}' est introuvable.")
        return

    # Nettoyage et Agrégation
    if 'entreprise' not in df.columns: df['entreprise'] = df.iloc[:, 0]
    df_clean = df.copy()
    for col in FEATURES: df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    df_clean = df_clean.dropna(subset=FEATURES)
    df_clean['Poids liquidités'] = df_clean['Poids liquidités'].clip(lower=0).clip(upper=1.0)
    df_clean["ratio d'endettement"] = df_clean["ratio d'endettement"].clip(upper=20.0)

    # 3.1. Préparation pour le Scaler et le Clustering
    df_company_profile = df_clean.groupby('entreprise')[FEATURES].mean().reset_index()
    X = df_company_profile[FEATURES].values

    if len(X) < K:
        raise ValueError(f"ERREUR : Seulement {len(X)} entreprises valides. Impossible de former {K} clusters.")

    # Normalisation et Clustering sur les MOYENNES D'ENTREPRISES
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=K, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(X_scaled)

    # 3.2. Récupération des tables de mapping et de résultats

    # Création de la table de mapping des Cluster ID -> Nom Qualitatif
    cluster_centers_unscaled = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=FEATURES)
    cluster_counts = df_company_profile.groupby(kmeans.labels_).size().rename('Nombre d\'entreprises')
    cluster_centers_unscaled = cluster_centers_unscaled.join(cluster_counts)
    df_mapping_profils = classify_profiles_unique(cluster_centers_unscaled)
    mapping_table = df_mapping_profils.set_index('Original_Cluster')['Profil_Qualitatif']

    # --- 3.3. SAUVEGARDE DES PROFILS D'ENTREPRISE MOYENS (Résultat standard) ---
    df_company_profile['Cluster'] = kmeans.labels_
    df_final_result = pd.merge(
        df_company_profile,
        mapping_table,
        left_on='Cluster',
        right_index=True,
        how='left'
    )
    df_profiles_output = df_final_result[['entreprise', 'Profil_Qualitatif'] + FEATURES].round(3)
    df_profiles_output.to_csv(PROFILES_RESULTS_PATH, index=False)

    # --- 3.4. CALCUL ET SAUVEGARDE DES PROFILS FEC INDIVIDUELS (NOUVEAU) ---

    # Normaliser chaque ligne FEC individuelle avec le scaler entraîné sur les moyennes
    X_fecs_scaled = scaler.transform(df_clean[FEATURES].values)

    # Prédire le Cluster ID pour chaque FEC individuel
    fec_labels = kmeans.predict(X_fecs_scaled)

    df_fecs_individuels = df_clean.copy()
    df_fecs_individuels['Cluster_ID'] = fec_labels

    # Joindre le Nom du Profil Qualitatif à chaque FEC
    df_fecs_individuels = pd.merge(
        df_fecs_individuels,
        mapping_table,
        left_on='Cluster_ID',
        right_index=True,
        how='left'
    ).rename(columns={'Profil_Qualitatif': 'Profil_FEC_Individuel'})

    # Sauvegarde des résultats individuels (avec le nom du FEC/année)
    df_fecs_individuels[['entreprise', 'Cluster_ID', 'Profil_FEC_Individuel'] + FEATURES].to_csv(FECS_PROFIL_INDIVIDUEL_PATH, index=False)

    # --- 3.5. Sauvegarde des Assets du Modèle (Scaler et Centres) ---

    with open(SCALER_PATH, 'wb') as file:
        pickle.dump(scaler, file)

    df_centres_final = cluster_centers_unscaled.join(mapping_table).reset_index().rename(columns={'index': 'Cluster_ID'})
    df_centres_final.to_csv(CENTRES_PATH, index=False)

    print(f"\n🎉 Entraînement et sauvegarde terminés (K={K}).")
    print(f" -> Profils individuels (FECs) enregistrés sous : '{FECS_PROFIL_INDIVIDUEL_PATH}'")
    print(f" -> Profils agrégés (Entreprises) enregistrés sous : '{PROFILES_RESULTS_PATH}'")

if __name__ == '__main__':
    train_model_and_save_assets()