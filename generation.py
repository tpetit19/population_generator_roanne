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

from model import *

class Generation:
    def __init__(self,
                 json_path,
                 output_path,
                 batch_size=5,
                 total_batches=40,
                 max_time_seconds=20):
        self.json_path = json_path
        self.batch_size = batch_size
        self.total_batches = total_batches
        self.output_path = output_path
        self.max_time_seconds = max_time_seconds
        self.existing = pd.DataFrame()

    def run(self, sep=','):
        print("\n=== LANCEMENT DE LA GÉNÉRATION PAR LOTS ===")

        for step in range(self.total_batches):
            print(f"[INFO] Étape {step + 1}: Génération de {self.batch_size} humains (extension d'une population de {len(self.existing)} / {self.batch_size*self.total_batches} individus)")
            if DEBUG:
                print(f"[DEBUG] Nombre d'existants avant résolution : {len(self.existing)}")
            #if not self.existing.empty:
                #print("[DEBUG] Solutions existantes (IDs) :")
                #print(self.existing)

            m = Model(self.json_path, self.batch_size, self.total_batches, existing_humans=self.existing)
            m.solve(max_time_seconds=self.max_time_seconds)

            if m.get_solution() is not None:
                #print(m.get_solver_solution_dataframe().iloc[-self.batch_size:])
                new_df = m.get_solver_solution_dataframe().iloc[-self.batch_size:]
                self.existing = pd.concat([self.existing, new_df], ignore_index=True)
                #print(self.existing)
            else:
                print("[ERROR] Aucune solution trouvée.")
                break

        if not self.existing.empty:
            readable_df = self.existing.copy()
            m_population = m.get_population()

            for var in readable_df.columns:
                if var in m_population.variables:
                    df = m_population.variables[var]
                    id_col, val_col = df.columns[0], df.columns[1]
                    mapping = dict(zip(df[id_col], df[val_col]))
                    readable_df[var] = readable_df[var].map(mapping)
            # mix th lines
            readable_df = readable_df.sample(frac=1, random_state=None).reset_index(drop=True)
            # save to CSV
            readable_df.to_csv(self.output_path, index=False, sep=sep)
            print(f"\n[INFO] Fichier de population écrit dans : {self.output_path}")
        else:
            print("[ERROR] Aucun humain généré, fichier non créé.")

class Test:
    @staticmethod
    def test_model_batch_generation():
        print("\n=== TEST DE GÉNÉRATION PAR LOTS ===")
        json_path = "oldies/json_francais/population_schema_bins2.json_francais"
        batch_size = 5
        total_batches = 30

        existing = pd.DataFrame()

        for step in range(total_batches):
            print(f"\n[INFO] Étape {step + 1}: Génération de {batch_size} humains")
            #print(f"[DEBUG] Nombre d'existants avant résolution : {len(existing)}")
            #if not existing.empty:
                #print("[DEBUG] Solutions existantes (IDs) :")
                #print(existing)

            m = Model(json_path, batch_size, existing_humans=existing)
            m.solve(max_time_seconds=20)

            if m.get_solution() is not None:
                new_df = m.get_solver_solution_dataframe().iloc[-batch_size:]  # uniquement les nouveaux
                existing = pd.concat([existing, new_df], ignore_index=True)
            else:
                print("[ERROR] Aucune solution trouvée.")
                break

        if not existing.empty:
            output_path = "oldies/output/test2.csv"
            readable_df = existing.copy()

            population = m.get_population()
            for var in readable_df.columns:
                if var in population.variables:
                    df = population.variables[var]
                    id_col, val_col = df.columns[0], df.columns[1]
                    mapping = dict(zip(df[id_col], df[val_col]))
                    readable_df[var] = readable_df[var].map(mapping)

            readable_df.to_csv(output_path, index=False)
            print(f"\n[INFO] Fichier de population écrit dans : {output_path}")
        else:
            print("[ERROR] Aucun humain généré, fichier non créé.")



    @staticmethod
    def test_population_loading():
        print("=== TEST DE CHARGEMENT DE POPULATION (NOUVEAU JSON) ===")

        # Chemin du JSON simplifié
        json_path = "oldies/json_francais/population_schema_bins2.json_francais"

        if not os.path.exists(json_path):
            print(f"[FAILED] Fichier {json_path} non trouvé.")
            return

        # Instanciation et chargement
        pop = Population()
        pop.load_from_json(json_path)

        # Vérifications
        expected_vars = ["MBTI", "DEPARTEMENT", "COMMUNE", "NOM", "AGE", "SURFACE_APPARTEMENT"]

        all_ok = True
        for var in expected_vars:
            if var not in pop.variables:
                print(f"[FAILED] Variable '{var}' absente.")
                all_ok = False
            else:
                print(f"[OK] Variable '{var}' présente avec {len(pop.variables[var])} éléments.")

        # Vérification de la dépendance COMMUNE -> DEPARTEMENT
        if ("COMMUNE", "DEPARTEMENT") not in pop.dependencies:
            print("[FAILED] Dépendance (COMMUNE -> DEPARTEMENT) absente.")
            all_ok = False
        else:
            nb_couples = len(pop.dependencies[("COMMUNE", "DEPARTEMENT")])
            print(f"[OK] Dépendance (COMMUNE -> DEPARTEMENT) avec {nb_couples} couples.")

        if all_ok:
            print("[SUCCESS] Toutes les vérifications sont passées.")
        else:
            print("[WARNING] Certaines vérifications ont échoué.")

    # NEW
    @staticmethod
    def test_validate_forbidden_tuples_constraints():
        print("\n=== TEST DE VALIDATION DES CONTRAINTES ===")
        try:
            json_path = "oldies/json_francais/example2.json_francais"  # Ton fichier JSON enrichi
            population = Population()
            population.load_from_json(json_path)
            population.add_forbidden_tuples_constraints()
            print("[SUCCESS] Validation des contraintes terminée.\n")
        except Exception as e:
            print(f"[FAILED] Erreur durant la validation des contraintes : {e}")


class AdHoc:
    @staticmethod
    def replace_salary_values(csv_path, output_path=None):
        """
        Remplace les valeurs de la colonne 'SALAIRE' dans un fichier CSV selon un mapping défini.

        Args:
            csv_path (str): chemin du fichier CSV source.
            output_path (str, optional): chemin de sauvegarde. Si None, remplace le fichier original.
        """
        try:
            df = pd.read_csv(csv_path)

            if "SALAIRE" not in df.columns:
                print("[ERROR] Colonne 'SALAIRE' absente dans le fichier.")
                return

            mapping = {
                1: "Sans emploi / Temps partiel",
                2: "Retraité",
                3: "1200 à 1699€ / mois",
                4: "1700 à 2999€ / mois",
                5: "3000 à 5999€ / mois ",
                6: "> 6000€ / mois"
            }

            df["SALAIRE"] = df["SALAIRE"].map(mapping).fillna(df["SALAIRE"])

            if output_path is None:
                output_path = csv_path  # Remplace le fichier d'origine

            df.to_csv(output_path, index=False)
            print(f"[INFO] Fichier sauvegardé : {output_path}")

        except Exception as e:
            print(f"[ERROR] Erreur lors du traitement : {e}")