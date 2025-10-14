# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Created by: tpetit@pollitics.com
# www.pollitics.com

import pandas as pd
import matplotlib.pyplot as plt
import unicodedata
import os
from pathlib import Path
from generation import Generation
import csv

SEPARATOR = ";"

def read_auto_sep(path, default=","):
    # Essaie d'inférer le séparateur
    with open(path, "r", encoding="utf-8") as f:
        sample = f.read(4096)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=[",",";","\t","|"])
            sep = dialect.delimiter
        except csv.Error:
            sep = default
    return pd.read_csv(path, sep=sep), sep

# Fonction pour normaliser les noms (remplacer les accents et espaces)
def normalize_string(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn').upper().replace(" ",
                                                                                                                  "")
def generate(json_dir ="json",
             source_dir ="source",
             # Paramètres de génération
             batch_size = 10,
             total_batches = 10,
             max_time_seconds = 20
             ):
    total = batch_size * total_batches
    output_dir = "output_"+str(total)
    # Créer le répertoire output s'il n'existe pas
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Lister tous les fichiers JSON dans json_francais/
    json_files = [f for f in os.listdir(json_dir) if f.endswith(".json")]

    if not json_files:
        print("[ERROR] Aucun fichier JSON trouvé dans le répertoire 'json/'.")
        exit(1)

    for json_file in json_files:
        country = os.path.splitext(json_file)[0]
        json_path = os.path.join(json_dir, json_file)
        output_subdir = os.path.join(output_dir, country)
        output_path = os.path.join(output_subdir, f"population_{country}.csv")
        output_path_f = os.path.join(output_subdir, f"population_{country}_f.csv")
        plot_path = os.path.join(output_subdir, f"distribution_plot_{country}.png")

        print(f"\n[INFO] Traitement du fichier: {json_file}")

        # Créer le sous-répertoire de sortie
        Path(output_subdir).mkdir(parents=True, exist_ok=True)

        # Étape 1: Génération de la population
        print(f"[INFO] Génération de la population pour {country}...")
        try:
            g = Generation(json_path, output_path, batch_size=batch_size, total_batches=total_batches,
                           max_time_seconds=max_time_seconds)
            g.run(sep=';')
        except Exception as e:
            print(f"[ERROR] Échec de la génération pour {country}: {e}")
            continue

        # Vérifier si le fichier de sortie existe
        if not os.path.exists(output_path):
            print(f"[ERROR] Fichier {output_path} non généré.")
            continue

        # Étape 2: Post-traitement
        '''
        print(f"[INFO] Post-traitement pour {country}...")
        try:
            df = pd.read_csv(output_path,sep=',')   # Generation uses ',' not ';'
            print(df.head(3))
        except FileNotFoundError:
            print(f"[ERROR] Fichier {output_path} non trouvé.")
            continue

        # Charger le fichier de correspondance (source/xxx.csv)
        source_path = os.path.join(source_dir, f"{country}.csv")
        try:
            mapping_df, source_sep = read_auto_sep(source_path)
            print(source_sep)
            #mapping_df = pd.read_csv(source_path,sep=';')
            print(mapping_df.head(3))
        except FileNotFoundError:
            print(f"[ERROR] Fichier {source_path} non trouvé.")
            continue

        # Créer un dictionnaire de correspondance index -> catégorie
        mapping = {}
        for _, row in mapping_df.iterrows():
            char = normalize_string(row["Caractéristique"])
            index = row["Index"]
            category = row["Catégorie"]
            if char not in mapping:
                mapping[char] = {}
            mapping[char][index] = category
            try:
                numeric_index = int(index.split("-")[0])
                mapping[char][numeric_index] = category
            except ValueError:
                print(f"[WARNING] Index invalide '{index}' pour la caractéristique '{char}' dans {source_path}.")

        # Remplacer les indices par les catégories
        for column in df.columns:
            if column in mapping:
                try:
                    df[column] = df[column].apply(lambda x: int(x) if pd.notnull(x) else x)
                except (ValueError, TypeError):
                    print(f"[WARNING] Certaines valeurs dans la colonne {column} ne sont pas convertibles en entiers.")
                df[column] = df[column].map(mapping.get(column, {}), na_action='ignore')
                unmapped_values = df[column][df[column].isna() & df[column].notna()].unique()
                if len(unmapped_values) > 0:
                    print(f"[WARNING] Valeurs non mappées dans la colonne {column}: {unmapped_values}")
            else:
                print(f"[WARNING] La colonne {column} n'a pas de correspondance dans {source_path}.")

        # Mélanger les lignes
        df = df.sample(frac=1, random_state=None).reset_index(drop=True)

        # Sauvegarder le CSV post-traité
        try:
            df.to_csv(output_path_f, index=False, sep=SEPARATOR)
            print(f"[INFO] Fichier post-traité généré: {output_path_f}")
        except Exception as e:
            print(f"[ERROR] Impossible de sauvegarder {output_path_f}: {e}")
            continue

        # Étape 3: Vérification (visualisation)
        print(f"[INFO] Génération du graphique pour {country}...")
        try:
            dist_df = pd.read_csv(source_path, sep=source_sep)  # réutilise le même sep
            pop_df = pd.read_csv(output_path_f, sep=SEPARATOR)
        except FileNotFoundError as e:
            print(f"[ERROR] Fichier non trouvé pour la vérification: {e}")
            continue

        # Liste des caractéristiques
        characteristics = dist_df["Caractéristique"].unique()
        char_mapping = {normalize_string(char): char for char in characteristics}
        columns = df.columns.tolist()

        # Créer une figure avec 2 lignes
        fig, axes = plt.subplots(2, len(columns), figsize=(20, 8), sharey="row")
        fig.suptitle(f"Distributions Attendues vs Réelles ({country})", fontsize=16)

        # Pour chaque caractéristique
        for idx, col in enumerate(columns):
            char_name = char_mapping.get(col, col)
            expected = dist_df[dist_df["Caractéristique"] == char_name][["Catégorie", "Distribution (2020-2024)"]]
            expected["Distribution (2020-2024)"] = expected["Distribution (2020-2024)"].str.replace("%", "").astype(float)
            categories = expected["Catégorie"].tolist()
            expected_percentages = expected["Distribution (2020-2024)"].tolist()

            total_count = len(pop_df)
            actual_counts = pop_df[col].value_counts(normalize=True) * 100
            actual_percentages = [actual_counts.get(cat, 0) for cat in categories]

            axes[0, idx].bar(categories, expected_percentages, color="blue", alpha=0.6)
            axes[0, idx].set_title(f"{char_name} (Attendu)")
            axes[0, idx].set_ylim(0, 100)
            axes[0, idx].set_ylabel("Pourcentage (%)" if idx == 0 else "")
            axes[0, idx].tick_params(axis="x", rotation=45)

            axes[1, idx].bar(categories, actual_percentages, color="orange", alpha=0.6)
            axes[1, idx].set_title(f"{char_name} (Réel)")
            axes[1, idx].set_ylim(0, 100)
            axes[1, idx].set_ylabel("Pourcentage (%)" if idx == 0 else "")
            axes[1, idx].tick_params(axis="x", rotation=45)

        # Ajuster l'espacement et sauvegarder le graphique
        try:
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.savefig(plot_path)
            plt.close()
            print(f"[INFO] Graphique généré: {plot_path}")
        except Exception as e:
            print(f"[ERROR] Impossible de sauvegarder le graphique {plot_path}: {e}")
            plt.close()
            continue
    '''

    print("[INFO] Traitement de tous les fichiers JSON terminé.")