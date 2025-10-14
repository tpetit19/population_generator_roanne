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

import json
import pandas as pd
import os
from ortools.sat.python import cp_model

DEBUG = False

# Population extracted from a JSON file
class Population:

    '''
    Creates the dataframes of the Population structure from a JSON definition file.

    self.variables:
    Dictionary where keys are variable names and values are dataframes with two columns:
    - The "VALEUR" column stores the value as used in the constraint solver's domain (indexed from 0).
    - The second column is named according to the "column" field in the JSON file and contains the raw item.

    self.dependencies:
    Dependency dictionary where:
    - Keys are pairs (variable, parent_variable) taken from the variable names in the JSON file, e.g., ("COMMUNE", "DEPARTEMENT").
    - The values are lists of tuples of index pairs (from the VALEUR field of the variable dataframes),
     defining the allowed tuples—no solution can include a tuple outside this list for the given variable pair.

    self.distributions:
    Dictionary where keys are variable names of type 'distribution_bins'.
    Each value is a list of tuples (bin_label, percentage), representing the target distribution specified
    in the JSON for that variable.
    Example: [('18-27', 18), ('28-37', 14), ...]

    self.variable_specs:
    Stores the original JSON specification for each variable. Keys are variable names, and values are
    dictionaries containing the fields from the JSON schema (type, file, column, distribution, etc.).

    self.forbidden_tuples:
    Dictionary where keys are tuples of variable names (e.g., ("AGE", "SALAIRE")) and values are lists of tuples.
    Each inner tuple is a combination of solver IDs (from self.variables) that are explicitly forbidden
    in the solution.

    self.allowed_tuples:
    Similar to self.forbidden_tuples, but for explicitly allowed combinations. If defined, only the tuples listed
    are permitted for the specified group of variables.
    '''


    # VARIABLES
    def __init__(self):
        # Initializes the Population object with empty structures for variables, dependencies, and constraints.

        self.variables = {}
        self.dependencies = {}
        self.distributions = {}
        self.variable_specs = None
        self.forbidden_tuples = {}
        self.allowed_tuples = {}

    def load_from_json(self, json_path):
        # Loads variable specifications and structures from a JSON configuration file.

        if not os.path.exists(json_path):
            print(f"[WARNING] JSON file '{json_path}' not found")
            return
        with open(json_path, "r") as f:
            config = json.load(f)
        self.variable_specs = config

        for name, spec in config.items():
            if spec["type"] == "independent":
                self.load_independent_variable(name, spec)
            elif spec["type"] == "dependent":
                self.load_dependent_variable(name, spec)
            elif spec["type"] == "distribution_bins":
                self.load_distribution_bins_variable(name, spec)
            elif spec["type"] == "independent_range":
                self.load_independent_range_variable(name, spec)
            elif spec["type"] == "constraint":
                if spec["subtype"] == "forbidden_tuples":
                    self.load_forbidden_tuples_constraints()
                elif spec["subtype"] == "allowed_tuples":
                    self.load_allowed_tuples_constraints()
                else:
                    pass
            else:
                print(f"[WARNING] Unknown type for variable or constraint '{name}'")

    def load_independent_variable(self, varname, spec):
        # Loads an independent variable from a CSV file and stores its values and solver domain.

        try:
            df = pd.read_csv(spec["file"], dtype=str)
            if DEBUG:
                print(df.head())
            col = spec["column"]
            sample = spec.get("sample", "__ALL__")

            if col not in df.columns:
                if DEBUG:
                    print(f"[WARNING] Column '{col}' not found in {spec['file']}")
                return

            df = df[[col]].dropna().drop_duplicates()
            df = df[~df[col].isin(["", "NaN", "nan"])].reset_index(drop=True)

            if sample != "__ALL__":
                df = df.sample(n=int(sample), random_state=42).reset_index(drop=True)

            df.insert(0, "VALEUR", df.index)
            self.variables[varname] = df.rename(columns={col: varname})
            if DEBUG:
                print(f"[INFO] Variable '{varname}' chargée avec {len(df)} lignes.")
                print(self.variables[varname])
        except Exception as e:
            print(f"[WARNING] Failed to load independent variable '{varname}': {e}")

    def load_dependent_variable(self, varname, spec):
        # Loads a dependent variable and builds a dependency table by matching keys to a parent variable.

        try:
            file_path = spec["file"]
            vcolumn = spec["vcolumn"]
            parent_varname = spec["var2"]
            key = spec["key"]
            key_in_var2 = spec["key_in_var2"]
            sample_per_value = spec.get("sample_per_value", "__ALL__")

            if parent_varname not in self.variables:
                if DEBUG:
                    print(f"[ERROR] Parent variable '{parent_varname}' must be loaded first.")
                return

            df_dep = pd.read_csv(file_path, dtype=str)
            df_dep = df_dep[[vcolumn, key]].dropna().drop_duplicates()

            # Reload parent CSV to get matching keys
            parent_spec = self.variable_specs[parent_varname]
            df_parent = pd.read_csv(parent_spec["file"], dtype=str)
            df_parent = df_parent[[key_in_var2, parent_spec["column"]]].dropna().drop_duplicates()

            # Build map: readable value -> key
            readable_to_key = dict(zip(df_parent[parent_spec["column"]], df_parent[key_in_var2]))

            # Parent VALEUR -> readable value
            id_to_readable = dict(zip(self.variables[parent_varname]["VALEUR"], self.variables[parent_varname][parent_varname]))

            # Build allowed keys based on sampled parent variables
            allowed_keys = set()
            for id_val in id_to_readable:
                readable_val = id_to_readable[id_val]
                if readable_val in readable_to_key:
                    allowed_keys.add(readable_to_key[readable_val])

            # Filter dependent DataFrame
            df_dep = df_dep[df_dep[key].isin(allowed_keys)]

            if df_dep.empty:
                if DEBUG:
                    print(f"[ERROR] Aucun élément trouvé pour construire la dépendance '{varname}'.")
                return

            # Sample per parent key
            final_rows = []
            for val in allowed_keys:
                group = df_dep[df_dep[key] == val]
                if sample_per_value != "__ALL__":
                    group = group.sample(n=min(int(sample_per_value), len(group)), random_state=42)  # tirage aléatoire
                final_rows.append(group)

            df_final = pd.concat(final_rows).drop_duplicates().reset_index(drop=True)
            df_final = df_final[[vcolumn, key]]

            # Build variable
            df_var = df_final[[vcolumn]].drop_duplicates().reset_index(drop=True)
            df_var.insert(0, "VALEUR", df_var.index)
            self.variables[varname] = df_var.rename(columns={vcolumn: varname})

            # Build dependency table
            vname_to_id = dict(zip(self.variables[varname][varname], self.variables[varname]["VALEUR"]))
            pname_to_id = dict(zip(self.variables[parent_varname][parent_varname], self.variables[parent_varname]["VALEUR"]))

            dep_pairs = []
            for _, row in df_final.iterrows():
                v_readable = row[vcolumn]
                p_key = row[key]
                # retrieve readable parent
                matches = [k for k, v in readable_to_key.items() if v == p_key]
                if matches:
                    p_readable = matches[0]
                    if v_readable in vname_to_id and p_readable in pname_to_id:
                        dep_pairs.append((vname_to_id[v_readable], pname_to_id[p_readable]))

            self.dependencies[(varname, parent_varname)] = dep_pairs
            if DEBUG:
                print(varname, parent_varname)
                print(self.dependencies[(varname, parent_varname)])
                print(f"[INFO] Variable dépendante '{varname}' chargée avec {len(dep_pairs)} éléments liés à '{parent_varname}'.")

        except Exception as e:
            print(f"[ERROR] Erreur lors du chargement de la dépendance '{varname}': {e}")

    def load_independent_range_variable(self, key, spec):
        # Creates an independent integer variable using a numerical range directly defined in the JSON.

        try:
            range_vals = spec.get("range", [])
            if len(range_vals) != 2:
                if DEBUG:
                    print(f"[WARNING] Mauvais format de range pour '{key}'.")
                return
            min_val, max_val = range_vals
            data = pd.DataFrame({
                "ID": range(max_val - min_val + 1),
                key: list(range(min_val, max_val + 1))
            })
            self.variables[key] = data
            if DEBUG:
                print(f"[INFO] Variable '{key}' chargée depuis range [{min_val}, {max_val}] ({len(data)} lignes).")
        except Exception as e:
            print(f"[WARNING] Erreur lors du chargement de '{key}' (independent_range): {e}")

    def load_distribution_bins_variable(self, varname, spec):
        # Initializes a variable with values generated from numerical bins for use with distribution constraints.

        try:
            bins = spec["bins"]
            distribution_p = spec["distribution_percentages"]

            total_percent = sum(distribution_p)
            if abs(total_percent - 100) > 0.99:  # Tolérance pour floats (99.01% OK)
                print(f"[WARNING] La distribution pour '{varname}' ne fait pas 100% : {total_percent}")
                return

            parsed_bins = []
            for b in bins:
                try:
                    start, end = map(int, b.strip().split("-"))
                    parsed_bins.append((start, end))
                except Exception as e:
                    if DEBUG:
                        print(f"[ERROR] Format de bin incorrect pour '{b}' dans '{varname}': {e}")
                    return

            min_val = min(start for start, _ in parsed_bins)
            max_val = max(end for _, end in parsed_bins)

            values = list(range(min_val, max_val + 1))
            df = pd.DataFrame({
                "ID": range(len(values)),
                varname: values
            })
            self.variables[varname] = df
            if DEBUG:
                print(f"[INFO] Variable '{varname}' initialisée avec {len(values)} valeurs uniques (distribution déclarée).")

        except Exception as e:
            print(f"[ERROR] Erreur lors du traitement de '{varname}' (distribution_bins) : {e}")

    def load_normal_distributed_integer_variable(self, varname, count, min_value, max_value):
        # Generates a synthetic variable using a normal distribution mapped to integer values.

        try:
            import numpy as np
            values = np.random.normal(loc=0, scale=1, size=count)
            values = (values - values.min()) / (values.max() - values.min())
            values = values * (max_value - min_value) + min_value
            values = np.round(values).astype(int)
            df = pd.DataFrame({
                "VALEUR": range(count),
                varname: values
            })
            self.variables[varname] = df
            if DEBUG:
                print(f"[INFO] Variable '{varname}' générée ({count} entiers de {min_value} à {max_value})")
        except Exception as e:
            print(f"[WARNING] Échec de génération pour '{varname}': {e}")

    # CONSTRAINTS
    def load_forbidden_tuples_constraints(self):
        # Loads and stores forbidden tuples from CSV files for variables declared as forbidden constraints.

        for name, spec in self.variable_specs.items():
            if spec.get("type") != "constraint":
                continue
            if spec.get("subtype") != "forbidden_tuples":
                continue

            var_list = spec.get("variables")
            file_path = spec.get("tuples_file")

            if not var_list or not file_path:
                print(f"[ERROR] Contrainte '{name}' mal définie : 'variables' ou 'tuples_file' manquant.")
                continue

            # Vérifie que toutes les variables sont bien chargées
            if not all(var in self.variables for var in var_list):
                print(f"[ERROR] Une ou plusieurs variables de '{name}' ne sont pas chargées.")
                continue

            try:
                df_tuples = pd.read_csv(file_path, dtype=str)
            except Exception as e:
                print(f"[ERROR] Échec de lecture du fichier pour '{name}' : {e}")
                continue

            # Vérifie que toutes les colonnes attendues sont bien dans le fichier
            if not all(var in df_tuples.columns for var in var_list):
                print(df_tuples.columns, var_list)
                print(f"[ERROR] Colonnes manquantes dans '{file_path}' pour '{name}'.")
                continue

            # On convertit les valeurs lisibles en IDs internes
            forbidden_set = []
            for idx, row in df_tuples.iterrows():
                try:
                    ids = []
                    for var in var_list:
                        val = row[var]
                        df = self.variables[var]
                        id_col, val_col = df.columns[0], df.columns[1]
                        val_to_id = dict(zip(df[val_col].astype(str), df[id_col]))

                        val_str = str(row[var]).strip()
                        if val_str not in val_to_id:
                            raise ValueError(f"[WARN] Valeur '{val_str}' inconnue pour la variable '{var}'.")
                        ids.append(val_to_id[val_str])

                    forbidden_set.append(tuple(ids))
                except Exception as e:
                    if DEBUG:
                        print(f"[WARN] Tuple invalide à la ligne {idx + 1} : {e}")
                    continue

            self.forbidden_tuples[tuple(var_list)] = forbidden_set
            if DEBUG:
                print(f"[OK] Contrainte '{name}' ajoutée : {len(forbidden_set)} tuples interdits pour {var_list}.")

    def load_allowed_tuples_constraints(self):
        # Loads and stores allowed tuples from CSV files for variables declared as allowed constraints.

        for name, spec in self.variable_specs.items():
            if spec.get("type") != "constraint":
                continue
            if spec.get("subtype") != "allowed_tuples":
                continue

            var_list = spec.get("variables")
            file_path = spec.get("tuples_file")

            if not var_list or not file_path:
                print(f"[ERROR] Contrainte '{name}' mal définie : 'variables' ou 'tuples_file' manquant.")
                continue

            # Vérifie que toutes les variables sont bien chargées
            if not all(var in self.variables for var in var_list):
                print(self.variables, var_list)
                print(f"[ERROR] Une ou plusieurs variables de '{name}' ne sont pas chargées.")
                continue

            try:
                df_tuples = pd.read_csv(file_path, dtype=str)
            except Exception as e:
                print(f"[ERROR] Échec de lecture du fichier pour '{name}' : {e}")
                continue

            # Vérifie que toutes les colonnes attendues sont bien dans le fichier
            if not all(var in df_tuples.columns for var in var_list):
                print(f"[ERROR] Colonnes manquantes dans '{file_path}' pour '{name}'.")
                continue

            # On convertit les valeurs lisibles en IDs internes
            allowed_set = []
            for idx, row in df_tuples.iterrows():
                try:
                    ids = []
                    for var in var_list:
                        val = row[var]
                        df = self.variables[var]
                        id_col, val_col = df.columns[0], df.columns[1]
                        val_to_id = dict(zip(df[val_col].astype(str), df[id_col]))

                        val_str = str(row[var]).strip()
                        if val_str not in val_to_id:
                            raise ValueError(f"[WARN] Valeur '{val_str}' inconnue pour la variable '{var}'.")
                        ids.append(val_to_id[val_str])

                    allowed_set.append(tuple(ids))
                except Exception as e:
                    if DEBUG:
                        print(f"[WARN] Tuple invalide à la ligne {idx + 1} : {e}")
                    continue

            self.allowed_tuples[tuple(var_list)] = allowed_set
            if DEBUG:
                print(f"[OK] Contrainte '{name}' ajoutée : {len(allowed_set)} tuples autorisés pour {var_list}.")

    # UTIL
    def get_variable_size(self, varname):
        # Returns the number of values available for a given variable (domain size).

        if varname in self.variables:
            return len(self.variables[varname])
        else:
            if DEBUG:
                print(f"[WARNING] Variable '{varname}' inconnue.")
            return None

    def display_summary(self):
        # Displays a summary of all loaded variables and defined dependencies.
        '''
        print("=== VARIABLES ===")
        for var, df in self.variables.items():
            print(f"{var}: {len(df)} IDs")
        print("\n=== DÉPENDANCES ===")
        for (v1, v2), links in self.dependencies.items():
            print(f"{v1} -> {v2}: {len(links)} couples")
        '''
        return

    def truncate_variable(self, varname, k):
        # Reduces a variable's domain to its first k values.

        if varname not in self.variables:
            if DEBUG:
                print(f"[WARNING] Variable '{varname}' non trouvée.")
            return
        original_length = len(self.variables[varname])
        self.variables[varname] = self.variables[varname].head(k).reset_index(drop=True)
        if DEBUG:
            print(f"[INFO] Variable '{varname}' tronquée à {k} lignes (au lieu de {original_length}).")

