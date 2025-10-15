
import unicodedata
import pandas as pd
import json
import os
from pathlib import Path

DEBUG = False

# Fonction pour normaliser les noms (remplacer les accents et espaces)
def normalize_string(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn').upper().replace(" ",
                                                                                                                  "")

# Répertoires
SOURCE_DIR = "source_communes"
JSON_DIR = "json_communes"
CONSTRAINT_DIR = "constraint_data"
DATA_DIR = "data"
SEPARATOR = ";"

# Créer le répertoire json_old2 s'il n'existe pas
Path(JSON_DIR).mkdir(parents=True, exist_ok=True)

# Vérifier l'existence de noms.csv
noms_file = os.path.join(DATA_DIR, "noms.csv")
if not os.path.exists(noms_file):
    print(f"[ERROR] Fichier {noms_file} non trouvé. Le champ NOM ne sera pas ajouté.")

# Lister tous les fichiers CSV dans source/
csv_files = [f for f in os.listdir(SOURCE_DIR) if f.endswith(".csv")]

for csv_file in csv_files:
    country = csv_file.replace(".csv", "")
    csv_path = os.path.join(SOURCE_DIR, csv_file)
    json_path = os.path.join(JSON_DIR, f"{country}.json")

    # Lire le fichier CSV
    try:
        df = pd.read_csv(csv_path, encoding="utf-8",sep=SEPARATOR)
    except Exception as e:
        print(f"[ERROR] Impossible de lire {csv_path}: {e}")
        continue

    # Vérifier les colonnes attendues
    expected_columns = ["Characteristic", "Category", "Distribution", "Index"]
    if not all(col in df.columns for col in expected_columns):
        print(f"[ERROR] Colonnes manquantes dans {csv_file}. Colonnes attendues: {expected_columns}")
        continue

    # Normaliser les pourcentages en flotants
    #df["Distribution"] = df["Distribution"].str.replace("%", "").astype(float)
    df["Distribution"] = df["Distribution"].astype(float)
    # Créer le dictionnaire JSON
    json_data = {}
    characteristics = df["Characteristic"].unique()
    normalized_chars = [normalize_string(char) for char in characteristics]

    for char in characteristics:
        char_normalized = normalize_string(char)
        char_data = df[df["Characteristic"] == char]
        bins = char_data["Index"].tolist()
        percentages = char_data["Distribution"].tolist()

        # Vérifier que les pourcentages somment à 100
        if round(sum(percentages),0) != 100:
            print(f"[WARNING] La somme des pourcentages pour {char} dans {csv_file} n'est pas 100%: {sum(percentages)}")

        json_data[char_normalized] = {
            "type": "distribution_bins",
            "bins": bins,
            "distribution_percentages": percentages,
            "distribution": "__NONE__"
        }

    # Ajouter le champ NOM si le fichier noms.csv existe
    if os.path.exists(noms_file):
        json_data["NOM"] = {
            "type": "independent",
            "file": "data/noms.csv",
            "column": "PRENOM",
            "sample": "__ALL__",
            "distribution": "__ALLDIFFERENT__"
        }
        print(f"[INFO] Champ NOM ajouté pour {country} depuis {noms_file}")

    # Ajouter les contraintes depuis constraint_data/ (racine et sous-répertoires spécifiques au pays)
    constraint_files = [f for f in os.listdir(CONSTRAINT_DIR) if f.startswith("constraint_") and f.endswith(".csv")]

    # Vérifier l'existence d'un sous-répertoire spécifique au pays
    country_constraint_dir = os.path.join(CONSTRAINT_DIR, country)
    country_constraint_files = []
    if os.path.exists(country_constraint_dir) and os.path.isdir(country_constraint_dir):
        country_constraint_files = [f for f in os.listdir(country_constraint_dir) if
                                    f.startswith("constraint_") and f.endswith(".csv")]
        print(
            f"[INFO] Contraintes spécifiques trouvées pour {country} dans {country_constraint_dir}: {country_constraint_files}")
    else:
        print(f"[INFO] Aucun sous-répertoire de contraintes spécifique pour {country}")

    # Traiter les contraintes à la racine
    for idx, constraint_file in enumerate(constraint_files):
        constraint_path = os.path.join(CONSTRAINT_DIR, constraint_file)
        try:
            constraint_df = pd.read_csv(constraint_path, encoding="utf-8")
            constraint_vars = constraint_df.columns.tolist()
            normalized_constraint_vars = [normalize_string(var) for var in constraint_vars]
            # Vérifier si toutes les variables de la contrainte sont dans les Characteristics
            if not all(var in normalized_chars for var in normalized_constraint_vars):
                print(
                    f"[INFO] Contrainte {constraint_file} ignorée pour {country}: variables {normalized_constraint_vars} non toutes présentes dans {normalized_chars}")
                continue
            constraint_name = f"CONTRAINTE_{constraint_file.replace('constraint_', '').replace('.csv', '').upper()}"
            json_data[constraint_name] = {
                "type": "constraint",
                "subtype": "forbidden_tuples",
                "variables": normalized_constraint_vars,
                "tuples_file": f"{CONSTRAINT_DIR}/{constraint_file}"
            }
        except Exception as e:
            print(f"[ERROR] Impossible de lire {constraint_file}: {e}")
            continue

    # Traiter les contraintes spécifiques au pays
    for idx, constraint_file in enumerate(country_constraint_files):
        constraint_path = os.path.join(country_constraint_dir, constraint_file)
        try:
            constraint_df = pd.read_csv(constraint_path, encoding="utf-8")
            constraint_vars = constraint_df.columns.tolist()
            normalized_constraint_vars = [normalize_string(var) for var in constraint_vars]
            # Vérifier si toutes les variables de la contrainte sont dans les Characteristics
            if not all(var in normalized_chars for var in normalized_constraint_vars):
                print(
                    f"[INFO] Contrainte spécifique {constraint_file} ignorée pour {country}: variables {normalized_constraint_vars} non toutes présentes dans {normalized_chars}")
                continue
            constraint_name = f"CONTRAINTE_{constraint_file.replace('constraint_', '').replace('.csv', '').upper()}"
            json_data[constraint_name] = {
                "type": "constraint",
                "subtype": "forbidden_tuples",
                "variables": normalized_constraint_vars,
                "tuples_file": f"{country_constraint_dir}/{constraint_file}"
            }
        except Exception as e:
            print(f"[ERROR] Impossible de lire {constraint_file} dans {country_constraint_dir}: {e}")
            continue

    # Écrire le fichier JSON
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=4, ensure_ascii=False)
        print(f"[INFO] Fichier JSON généré: {json_path}")
    except Exception as e:
        print(f"[ERROR] Impossible d'écrire {json_path}: {e}")

print("[INFO] Génération des fichiers JSON terminée.")

json_files = [f for f in os.listdir(JSON_DIR) if f.endswith(".json")]

for json_file in json_files:
    json_path = os.path.join(JSON_DIR, json_file)
    country = json_file.replace(".json", "")
    print(f"[INFO] Vérification du fichier JSON: {json_path}")

    # Lire le fichier JSON
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"[ERROR] Syntaxe JSON invalide dans {json_file}: {e}")
        continue
    except Exception as e:
        print(f"[ERROR] Impossible de lire {json_file}: {e}")
        continue

    # Vérifier les Characteristics de type distribution_bins
    for char, data in json_data.items():
        if data.get("type") == "distribution_bins":
            # Vérifier que bins et distribution_percentages existent et ont la même longueur
            if "bins" not in data or "distribution_percentages" not in data:
                print(f"[ERROR] Champ 'bins' ou 'distribution_percentages' manquant pour {char} dans {json_file}")
                continue
            if len(data["bins"]) != len(data["distribution_percentages"]):
                print(
                    f"[ERROR] Longueur différente entre bins ({len(data['bins'])}) et distribution_percentages ({len(data['distribution_percentages'])}) pour {char} dans {json_file}")
                continue

            # Vérifier que les pourcentages sont des entiers non négatifs
            percentages = data["distribution_percentages"]
            '''
            if not all(isinstance(p, int) and p >= 0 for p in percentages):
                print(f"[ERROR] Pourcentages non entiers ou négatifs pour {char} dans {json_file}: {percentages}")
                continue
            '''
            if not all(p >= 0 for p in percentages):
                print(f"[ERROR] Pourcentages négatifs pour {char} dans {json_file}: {percentages}")
                continue

            # Vérifier que la somme des pourcentages vaut 100
            total = sum(percentages)
            if round(total,0) != 100:
                print(f"[ERROR] La somme des pourcentages pour {char} dans {json_file} n'est pas 100: {total}")
            else:
                if DEBUG:
                    print(f"[INFO] Somme des pourcentages correcte pour {char} dans {json_file}: {total}")

        elif data.get("type") == "independent" and char == "NOM":
            # Vérifier la structure du champ NOM
            expected_nom = {
                "type": "independent",
                "file": "data/noms.csv",
                "column": "PRENOM",
                "sample": "__ALL__",
                "distribution": "__ALLDIFFERENT__"
            }
            if data != expected_nom:
                print(f"[ERROR] Structure incorrecte pour NOM dans {json_file}: {data}")
            else:
                print(f"[INFO] Structure correcte pour NOM dans {json_file}")

        elif data.get("type") == "constraint":
            # Vérifier que les variables de la contrainte existent dans les Characteristics
            normalized_chars = [k for k, v in json_data.items() if v.get("type") == "distribution_bins"]
            if not all(var in normalized_chars for var in data.get("variables", [])):
                print(
                    f"[ERROR] Variables de contrainte {data.get('variables')} non présentes dans les Characteristics de {json_file}")
            # Vérifier que le fichier de tuples existe
            tuples_file = data.get("tuples_file")
            if tuples_file and not os.path.exists(tuples_file):
                print(f"[ERROR] Fichier de tuples {tuples_file} non trouvé pour la contrainte {char} dans {json_file}")
            else:
                if DEBUG:
                    print(f"[INFO] Contrainte {char} valide dans {json_file}")

print("[INFO] Post-vérification des fichiers JSON terminée.")