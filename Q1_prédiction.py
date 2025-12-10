import pandas as pd
import numpy as np
import pickle
from scipy.spatial.distance import cdist
import os
import sys

# --- 1. CONFIGURATION ET CHEMINS D'ACCÈS ---
FEATURES = ["ratio d'endettement", "Poids liquidités", "Poids provisions",
            "ratio découvert", "rotation des stock"]

# FICHIERS DE MODÈLE ENTRÉE (Doivent exister après l'exécution de train.py)
CENTRES_PATH = 'model_centres.csv'
SCALER_PATH = 'model_scaler.pkl'
NEW_FEC_PATH = '000044.csv' # REMPLACEZ PAR LE NOM DE VOTRE NOUVEAU FEC

# --- 2. FONCTIONS DE CALCUL DES RATIOS ---
# Ces fonctions sont cruciales pour transformer le nouveau FEC brut en ratios

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
    return ratio_endettement

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
    return Poids_Liquidité

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
    return Decouvert

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
    return Poids_provisions

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
    return Rotation_stock


def calculate_indices(df):
    """ Crée un DataFrame d'UNE LIGNE à partir des valeurs scalaires. """
    for col in ["Debit", "Credit"]:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).replace ( "," , "." )

    data = {
        "ratio d'endettement": [ratio_endettement(df)],
        "Poids liquidités": [Poids_Liquidité(df)],
        "Poids provisions": [Poids_provisions(df)],
        "ratio découvert": [Decouvert(df)],
        "rotation des stock": [Rotation_stock(df)]
    }
    return pd.DataFrame(data)

# --- 3. FONCTION PRINCIPALE : CHARGEMENT ET CLASSIFICATION ---

def classify_new_fec():
    """ Charge le modèle sauvegardé et classifie le nouveau FEC. """

    print(f"\n======== DÉBUT DE LA CLASSIFICATION DE {NEW_FEC_PATH} ========")

    # 1. Chargement du Modèle
    try:
        # Chargement du Scaler (Règles de normalisation)
        with open(SCALER_PATH, 'rb') as file:
            scaler = pickle.load(file)

        # Chargement des Centres de Clusters (Profils)
        df_centres = pd.read_csv(CENTRES_PATH)

    except FileNotFoundError as e:
        print(f"ERREUR FATALE: Fichier modèle non trouvé : {e.filename}")
        print("Veuillez vous assurer d'avoir exécuté le script 'train.py' d'abord.")
        return
    except Exception as e:
        print(f"Erreur lors du chargement du modèle : {e}")
        return

    # 2. Calcul des Ratios du Nouveau FEC
    try:
        df_fec_new = pd.read_csv(NEW_FEC_PATH)
    except FileNotFoundError:
        print(f"ERREUR : Le nouveau FEC '{NEW_FEC_PATH}' est introuvable. Classification annulée.")
        return

    new_ratios_df = calculate_indices(df_fec_new)

    # Application des règles de nettoyage/capping/imputation (Doit être identique à l'entraînement)
    new_ratios_df['Poids liquidités'] = new_ratios_df['Poids liquidités'].clip(lower=0).clip(upper=1.0)
    new_ratios_df["ratio d'endettement"] = new_ratios_df["ratio d'endettement"].clip(upper=20.0)
    new_ratios_df = new_ratios_df.fillna(0)

    # 3. Prédiction
    X_new = new_ratios_df[FEATURES].values

    # Normalisation par le SCALER CHARGÉ
    X_new_scaled = scaler.transform(X_new)

    # Prédiction du cluster ID (Normalisation des centres par le même scaler)
    centres_unscaled = df_centres[FEATURES].values
    centres_scaled = scaler.transform(centres_unscaled)

    # Calcul de la distance euclidienne
    distances = cdist(X_new_scaled, centres_scaled, metric='euclidean')
    predicted_cluster_id = np.argmin(distances, axis=1)[0]

    # 4. Récupération du Profil Final
    predicted_profile = df_centres[df_centres['Cluster_ID'] == predicted_cluster_id]['Profil_Qualitatif'].iloc[0]

    print("\n" + "="*80)
    print(f"| RÉSULTAT DU PROFIL POUR LE NOUVEAU FEC : {os.path.basename(NEW_FEC_PATH)} |")
    print("="*80)
    print(f"CLASSEMENT : {predicted_profile}")
    print(f"Cluster ID : {predicted_cluster_id}")
    print("\nIndicateurs du nouveau FEC :")
    print(new_ratios_df.round(3).to_string(index=False))
    print("="*80)

if __name__ == '__main__':
    classify_new_fec()


if __name__ == '__main__':
    classify_new_fec()