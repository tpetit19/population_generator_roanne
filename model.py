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

from population import *
from ortools.sat.python import cp_model
import pandas as pd

DEBUG = False
STATS = True

class Model:

    def __init__(self, json_path, batch_size, nb_batches, existing_humans=None):
        """
        Initialise un modèle OR-Tools à partir d'un schéma de population JSON.
        """
        self.batch_size = batch_size
        self.nb_batches = nb_batches
        if isinstance(existing_humans, pd.DataFrame):
            self.existing_humans = existing_humans
        else:
            self.existing_humans = None
        self.model = cp_model.CpModel()
        self.population = Population()
        self.population.load_from_json(json_path)
        self.population.display_summary()
        self.json_path = json_path
        self.variables = {}
        # Variables
        self.build_variables()
        # Contraintes
        self.add_dependency_constraints()
        self.add_diversity_constraints()
        self.add_alldifferent_constraints()
        self.add_distribution_constraints()
        self.add_forbidden_tuples_constraints()
        self.add_allowed_tuples_constraints()
        # Solve
        self.solver = None
        self.solver_solution_dataframe = None
        self.solution = None

    def build_variables(self):
        """
        Crée les variables OR-Tools à partir des domaines définis dans Population.
        Si des humains préexistants sont fournis (self.existing_humans), ils sont ajoutés en tant que constantes.
        """
        self.variables = {}

        # Nombre total d'humains à générer (nouveaux + existants)
        n_existing = 0
        if isinstance(self.existing_humans, pd.DataFrame):
            n_existing = len(self.existing_humans)

        total_n = self.batch_size + n_existing
        #print(self.batch_size)

        for var_name, df in self.population.variables.items():
            if df.shape[1] < 2:
                #if DEBUG:
                print(f"[WARNING] Variable '{var_name}' mal formée.")
                continue
            '''
            domain_vals = []
            if df.shape[1] >= 2 and df.columns[1] != "ID":
                domain_vals = df.iloc[:, 1].dropna().unique()
            else:
                domain_vals = df.iloc[:, 0].dropna().unique()
            lb = int(domain_vals.min())
            ub = int(domain_vals.max())
            print(f"[DEBUG] Domaine de {var_name} : [{lb}, {ub}] sur {len(self.population.variables.items())} variables.")
            '''
            domain_ids = df.iloc[:, 0].dropna().unique()
            lb = int(domain_ids.min())
            ub = int(domain_ids.max())
            #print(f'domain: [{lb},{ub}]')

            initial_values = []

            # Ajouter les variables déjà assignées si self.existing_humans est bien fournie
            if (
                    isinstance(self.existing_humans, pd.DataFrame)
                    and var_name in self.existing_humans.columns
            ):
                for i in range(n_existing):
                    val = int(self.existing_humans.iloc[i][var_name])
                    initial_values.append(self.model.NewConstant(val))

            nb_new = total_n - len(initial_values)

            generated_vars = [
                self.model.NewIntVar(lb, ub, f"{var_name}_{i}") for i in range(nb_new)
            ]

            self.variables[var_name] = initial_values + generated_vars
            if DEBUG:
                print(f"[INFO] Variable OR-Tools '{var_name}' créée avec domaine [{lb}, {ub}] sur {self.batch_size} individus.")

    def add_dependency_constraints(self):
        """
        Ajoute les contraintes de dépendance fonctionnelle pour chaque couple (source → cible)
        défini dans self.population.dependencies.
        """
        for (source, target), allowed_pairs in self.population.dependencies.items():
            if source not in self.variables or target not in self.variables:
                if DEBUG:
                    print(f"[WARNING] Variables '{source}' ou '{target}' manquantes dans le modèle.")
                continue

            source_vars = self.variables[source]
            target_vars = self.variables[target]

            if len(source_vars) != len(target_vars):
                if DEBUG:
                    print(f"[WARNING] Taille incohérente entre {source} et {target}.")
                continue

            for i in range(len(source_vars)):
                self.model.AddAllowedAssignments(
                    [source_vars[i], target_vars[i]],
                    allowed_pairs
                )

            if DEBUG:
                print(f"[INFO] Contraintes posées pour {source} → {target} ({len(allowed_pairs)} couples).")

    def add_diversity_constraints(self):
        """
        Ajoute des contraintes de diversité pour les variables déclarées comme "__DIVERSE__"
        dans le schéma JSON (variables indépendantes avec "distribution" et variables dépendantes avec "var1_distribution").
        """
        self.diversity_bools = []
        n = self.batch_size + len(self.existing_humans) if self.existing_humans is not None else self.batch_size
        #print("EXISTING:",self.existing_humans)
        # Contraintes pour les variables indépendantes
        for var_name, spec in self.population.variable_specs.items():
            if spec.get("distribution", "__NONE__") == "__DIVERSE__" and var_name in self.variables:
                var_list = self.variables[var_name]
                if DEBUG:
                    print(f"[DIVERSE] Ajout de diversité sur variable '{var_name}'")
                for i in range(n):
                    for j in range(i + 1, n):
                        b = self.model.NewBoolVar(f"{var_name}_diff_{i}_{j}")
                        self.model.Add(var_list[i] != var_list[j]).OnlyEnforceIf(b)
                        self.model.Add(var_list[i] == var_list[j]).OnlyEnforceIf(b.Not())
                        self.diversity_bools.append(b)

        # Contraintes pour les variables dépendantes (var1)
        for var_name, spec in self.population.variable_specs.items():
            if spec.get("type") == "dependent_pair":
                if spec.get("var1_distribution", "__NONE__") == "__DIVERSE__":
                    var1 = spec["var1_name"]
                    if var1 in self.variables:
                        var_list = self.variables[var1]
                        if DEBUG:
                            print(f"[DIVERSE] Ajout de diversité sur variable dépendante '{var1}'")
                        for i in range(n):
                            for j in range(i + 1, n):
                                b = self.model.NewBoolVar(f"{var1}_diff_{i}_{j}")
                                self.model.Add(var_list[i] != var_list[j]).OnlyEnforceIf(b)
                                self.model.Add(var_list[i] == var_list[j]).OnlyEnforceIf(b.Not())
                                self.diversity_bools.append(b)
        '''
        if self.diversity_bools:
            print(f"[DIVERSE] Maximisation des différences sur {len(self.diversity_bools)} couples.")
            self.model.Maximize(sum(self.diversity_bools))
        else:
            print("[DIVERSE] Aucune contrainte de diversité définie dans le JSON.")
        '''

    def add_alldifferent_constraints(self):
        """
        Applique la contrainte AllDifferent aux variables définies comme "__ALLDIFFERENT__" dans le JSON.
        """
        count = 0
        for var_name, spec in self.population.variable_specs.items():
            dist_type = spec.get("distribution", "__NONE__")
            if dist_type == "__ALLDIFFERENT__":
                var_list = self.variables.get(var_name, [])
                if len(var_list) > 1:
                    self.model.AddAllDifferent(var_list)
                    if DEBUG:
                        print(
                        f"[ALLDIFFERENT] Contrainte AllDifferent appliquée à la variable '{var_name}' ({len(var_list)} individus).")
                    count += 1
                else:
                    if DEBUG:
                        print(
                        f"[ALLDIFFERENT] Trop peu de variables pour AllDifferent sur '{var_name}', contrainte ignorée.")
        if count == 0:
            if DEBUG:
                print("[ALLDIFFERENT] Aucune contrainte AllDifferent trouvée dans le JSON.")

    def add_distribution_constraints(self):
        """
        Ajoute des contraintes de distribution sur les variables de type "distribution_bins".
        Chaque bin a une cible (nombre d’individus attendus), calculée à partir du total (existants + batch).
        On pénalise l’écart absolu entre le nombre d’individus dans le batch et ce qu’il faudrait pour atteindre la cible.
        """
        self.distribution_penalties = []

        for var_name, spec in self.population.variable_specs.items():
            if spec.get("type") != "distribution_bins":
                continue

            bins = spec["bins"]
            percentages = spec["distribution_percentages"]
            parsed_bins = [tuple(map(int, b.split("-"))) for b in bins]
            # passage en IDs (déjà en ids!!)

            minv = min(parsed_bins[0])
            for i in parsed_bins:
                minv = min(i[0],minv)
            for i in range(len(parsed_bins)):
                parsed_bins[i] = (parsed_bins[i][0]-minv, parsed_bins[i][1]-minv)

            # ---
            #print(parsed_bins)
            current_vars = self.variables.get(var_name, [])
            current_size = len(current_vars)

            # Étape 1 : compter les individus déjà présents dans chaque bin
            existing_counts = [0] * len(parsed_bins)
            if isinstance(self.existing_humans, pd.DataFrame) and var_name in self.existing_humans.columns:
                for val in self.existing_humans[var_name]:
                    val = int(val)
                    #print("v = ",val)
                    for i, (start, end) in enumerate(parsed_bins):
                        if start <= val <= end:
                            existing_counts[i] += 1
                            break
            #print("E = ",existing_counts)

            # Étape 2 : calculer les cibles (nombre total attendu par bin)
            total_n = self.batch_size*self.nb_batches #sum(existing_counts) + batch_size
            float_ideals = [total_n * p / 100 for p in percentages]
            floored_ideals = [int(x) for x in float_ideals]
            remainder = total_n - sum(floored_ideals)

            # Répartir le reste (méthode des plus forts restes)
            fractions = sorted(
                [(i, float_ideals[i] - floored_ideals[i]) for i in range(len(bins))],
                key=lambda x: -x[1]
            )
            for i in range(remainder):
                floored_ideals[fractions[i][0]] += 1
            #print("floored ideals:",floored_ideals)
            #print(parsed_bins)

            # Étape 3 : contraintes pour chaque bin
            total_in_bin_array = []
            for i, (start, end) in enumerate(parsed_bins):
                #print("i =",i, "(start, end) =", (start, end))
                bin_bools = []
                for j, var in enumerate(current_vars):
                    #print("j, var", j, var)
                    # ignore constants as they are already counted
                    '''
                    domain = var.Proto().domain
                    if len(domain) == 2 and domain[0] == domain[1]:
                        print(f"Ignoring constant variable {j} = {domain[0]}")
                        continue
                    '''
                    #
                    b = self.model.NewBoolVar(f"{var_name}_in_bin_{i}_var_{j}")
                    self.model.Add(var >= start).OnlyEnforceIf(b)
                    self.model.Add(var <= end).OnlyEnforceIf(b)
                    # Gestion explicite des cas hors bin
                    is_lt = self.model.NewBoolVar(f"{var_name}_lt_{i}_{j}")
                    is_gt = self.model.NewBoolVar(f"{var_name}_gt_{i}_{j}")
                    self.model.Add(var < start).OnlyEnforceIf(is_lt)
                    self.model.Add(var >= start).OnlyEnforceIf(is_lt.Not())
                    self.model.Add(var > end).OnlyEnforceIf(is_gt)
                    self.model.Add(var <= end).OnlyEnforceIf(is_gt.Not())
                    self.model.AddBoolOr([is_lt, is_gt]).OnlyEnforceIf(b.Not())
                    bin_bools.append(b)

                # Nombre total d’individus dans le batch pour ce bin
                total_in_bin = self.model.NewIntVar(0, current_size, f"count_{var_name}_bin_{i}")
                self.model.Add(total_in_bin == sum(bin_bools))   # only sur les variables du batch courant donc ok
                total_in_bin_array.append(total_in_bin)

                # Cible restante à atteindre (ce que le batch devrait essayer de combler)
                target_total = floored_ideals[i]
                #already_there = existing_counts[i]
                #print(already_there)
                #print("\nalready_there = ",already_there)
                #print(already_there, end="")
                target_remaining = target_total #- already_there
                #print("\ntarget = ")
                #print(target_total, already_there, target_remaining)
                #print()
                # Pénalité = écart absolu entre nombre généré et cible restante
                diff = self.model.NewIntVar(-total_n, total_n, f"diff_{var_name}_{i}")
                abs_diff = self.model.NewIntVar(0, total_n, f"absdiff_{var_name}_{i}")
                self.model.Add(diff == (total_in_bin - target_remaining))
                self.model.AddAbsEquality(abs_diff, diff)
                self.distribution_penalties.append(abs_diff)
            self.model.Add(sum(total_in_bin_array) == current_size)   # importante pour l'efficacite

        #print("Contraintes de dstribution posées")

    def add_forbidden_tuples_constraints(self):
        """
        Ajoute les contraintes d'interdiction de combinaisons (tuples) définies dans self.population.forbidden_tuples.
        """
        if not hasattr(self.population, "forbidden_tuples"):
            if DEBUG:
                print("[INFO] Aucune contrainte de tuples interdits trouvée.")
            return

        for varnames, forbidden_list in self.population.forbidden_tuples.items():
            if not all(v in self.variables for v in varnames):
                if DEBUG:
                    print(f"[WARNING] Variables manquantes dans le modèle pour la contrainte : {varnames}")
                continue

            # On récupère la liste de variables OR-Tools pour chaque varname
            var_lists = [self.variables[v] for v in varnames]
            n = len(var_lists[0])
            if not all(len(vlist) == n for vlist in var_lists):
                print(f"[ERROR] Taille incohérente des variables dans la contrainte {varnames}")
                continue

            # Pour chaque individu i, on applique la contrainte sur les variables concernées
            for i in range(n):
                current_vars = [vlist[i] for vlist in var_lists]
                self.model.AddForbiddenAssignments(current_vars, forbidden_list)
            if DEBUG:
                print(f"[OK] Contrainte de table interdite appliquée pour {varnames} ({len(forbidden_list)} tuples).")

    def add_allowed_tuples_constraints(self):
        """
        Ajoute les contraintes de type 'allowed_tuples' à partir du dictionnaire
        self.population.allowed_tuples contenant :
          - clés : tuples de noms de variables
          - valeurs : listes de tuples autorisés (ID internes)
        """
        if not hasattr(self.population, "allowed_tuples") or not self.population.allowed_tuples:
            if DEBUG:
                print("[INFO] Aucune contrainte de tuples autorisés trouvée.")
            return

        for var_names, allowed in self.population.allowed_tuples.items():
            # Récupération des listes de variables OR-Tools
            try:
                variable_lists = [self.variables[v] for v in var_names]
            except KeyError as e:
                print(f"[ERROR] Variable manquante pour contrainte de tuples autorisés : {e}")
                continue

            # Transposition en une liste de tuples de variables (une par individu)
            for i in range(self.batch_size):
                tuple_vars = [var_list[i] for var_list in variable_lists]
                self.model.AddAllowedAssignments(tuple_vars, allowed)
            if DEBUG:
                print(
                f"[ALLOWED] Contrainte de table autorisée posée sur {var_names} ({len(allowed)} combinaisons autorisées).")

    def get_model(self):
        return self.model

    def get_population(self):
        return self.population

    def get_json_path(self):
        return self.json_path

    def get_variables(self):
        return self.variables

    def get_solver_solution_dataframe(self):
        return self.solver_solution_dataframe

    def get_solution(self):
        return self.solution

    def solve(self, max_time_seconds=None, stop_at_first_solution=False):
        self.solver = cp_model.CpSolver()
        self.solver.parameters.num_search_workers = 1
        # Objectif global : combiner diversité et distribution
        lenobj = 0
        # In solve() method:
        if hasattr(self, "diversity_bools") and hasattr(self, "distribution_penalties") and self.diversity_bools and self.distribution_penalties:
            print('BOTH')
            #TODO
            # Weight diversity higher than distribution
            self.model.Maximize(sum(self.diversity_bools) - sum(self.distribution_penalties))
            lenobj = len(self.diversity_bools) + len(self.distribution_penalties)
        elif hasattr(self, "diversity_bools") and self.diversity_bools and not self.distribution_penalties:
            print('DIVERSITY ONLY')
            self.model.Maximize(sum(self.diversity_bools))
            lenobj = len(self.diversity_bools)
        elif hasattr(self, "distribution_penalties") and self.distribution_penalties:
            print('DISTRIBUTION ONLY')
            self.model.Minimize(sum(self.distribution_penalties))
            lenobj = len(self.distribution_penalties)

        # end
        if max_time_seconds is not None:
            self.solver.parameters.max_time_in_seconds = max_time_seconds
        if stop_at_first_solution:
            self.solver.parameters.enumerate_all_solutions = False
            self.solver.parameters.solution_pool_size = 15
            self.solver.parameters.stop_after_first_solution = True
        #if DEBUG:
        #print(f"[DEBUG] Objectif basé sur {lenobj} variables.",self.distribution_penalties)
        status = self.solver.Solve(self.model)

        if status in [cp_model.FEASIBLE, cp_model.OPTIMAL]:
            if STATS:
                print("Number of conflicts (backtracks):", self.solver.NumConflicts())
            rows = []
            var_names = sorted(self.variables.keys())

            total_n = len(next(iter(self.variables.values())))

            for i in range(total_n):
                row = {}
                for var_name in var_names:
                    var_list = self.variables[var_name]
                    if i < len(var_list):
                        row[var_name] = self.solver.Value(var_list[i])
                rows.append(row)

            #print(rows)
            self.solver_solution_dataframe = pd.DataFrame(rows)
            self.solution = self.solver_solution_dataframe.copy()

            for var in self.solution.columns:
                if var in self.population.variables:
                    df = self.population.variables[var]
                    id_col = df.columns[0]
                    val_col = df.columns[1]
                    mapping = dict(zip(df[id_col], df[val_col]))
                    self.solution[var] = self.solution[var].map(mapping)
            #print(f"[INFO] Solution trouvée avec {len(self.solution)} individus.")
            sol = self.solution.values
            #print()
            #for v in self.solution.values:
            #    print(v[0],end=" ")
            #print()
        else:
            print("[ERROR] No solution found.")
            self.solution = None


def save_solution(self, path):
    if self.solution is None:
        print("[ERROR] Aucune solution disponible à enregistrer.")
    else:
        self.solution.to_csv(path, index=False)
        print(f"[INFO] Solution enregistrée dans {path}")
