import os
import sys
import random
import logging
import itertools
import itertools
import numpy as np
from math import comb
import scipy.constants as sp
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from numpy.core.numeric import isscalar
from scipy.optimize import least_squares

from typing import List, Tuple, Any

# Libraries for genetic algorithm
from genetic_algorithm import generate_new_mechanism
from genetic_algorithm import generate_mutations

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

pathname = os.path.dirname(sys.argv[0])
fullpath = os.path.abspath(pathname)


class ReactionMechanismGenerator():
    '''Class containing the reaction mechanism generator.'''
    def __init__(
        self,
        order=1, # order of the system, provided when creating the object, else 1
        include_bias=True, # whether to include sources and sinks
        tol=1e-5,
        max_ratio = 5, # maximum difference between the log of the lowest and highest k
        num_generations=1, # number of generations
        max_stoichiometric_ratio = 2, # maximum allowed stoichiometric ration (p/r)
        num_mech_per_gen = 100, # number of mechanisms per generation
        max_rxns_per_mech = 5, # maximum reactions per mechanisms
        min_rxns_per_mech = 5, # minimum reactions per mechanisms
        from_previous_generation=0.2, # fraction of previous generation to pass on to next generation
        mutation_prob = 0.1, # fraction of the individuals of a given generation that could go thru mutations

    ):

        # Input validation
        if order < 0:
            raise ValueError("order cannot be negative")
        if max_ratio < 1:
            raise ValueError("max_ratio cannot be less than 1")
        if num_generations < 1:
            raise ValueError("num_generations cannot be less than 1")
        if max_stoichiometric_ratio < 0:
            raise ValueError("max_stoichiometric_ratio cannot be negative")
        if num_mech_per_gen < 1:
            raise ValueError("num_mech_per_gen cannot be less than 1")
        if max_rxns_per_mech < min_rxns_per_mech:
            raise ValueError("rxns_per_mech cannot be less than min_rxns_per_mech")
        if not (1 <= min_rxns_per_mech <= max_rxns_per_mech):
            raise ValueError("rxns_per_mech cannot be less than 1")
        if not (0 < from_previous_generation <= 1):
            raise ValueError("from_previous_generation  ∈ [0,1]")
        if mutation_prob < 0:
            raise ValueError("mutation_prob ∈ [0,1]")
        if round(2*from_previous_generation*num_mech_per_gen) == 0:
            raise ValueError(f"from_previous_generation and num_mech_per_gen has been set such that no best_mechs has lenght 0")

        # Initialize parameters
        self.include_bias = include_bias
        self.ORDER = order
        self.prev_gen = from_previous_generation
        self.num_gens = num_generations
        self.TOL = tol
        self.MAX_RATIO = max_ratio # max ratio of largest coeff / threshold
        self.MAX_STOICH = max_stoichiometric_ratio
        self.NUM_MECHS = num_mech_per_gen
        self.MAX_RXN = max_rxns_per_mech
        self.MIN_RXN = min_rxns_per_mech
        self.P_MUTATION = mutation_prob
        self.N_TARGETS = 0
        self.reaction_matrix = []
        self.reaction_constants = []
        self.mse = 0


    def _generate_reactions(self) -> np.ndarray:
        """
        Generate all possible reactions for a given number of target species with associated probabilities.

        Returns
        -------
        np.ndarray
            An array of tuples, where each tuple contains a reaction (as a numpy array)
            and its associated probability (as a float).

        Raises
        ------
        ValueError
            If the sum of assigned probabilities does not equal 1 within the specified tolerance.
        """


        # Generate combinations for the reactants
        reactants_combinations = []
        for combination in itertools.product(range(self.ORDER + 1), repeat=self.N_TARGETS):
            if sum(combination) <= self.ORDER:
                reactants_combinations.append(combination)
        reactants_combinations = np.array(reactants_combinations)

        # Generate combinations for the products
        products_combinations = []
        for combination in itertools.product(range(self.ORDER * self.MAX_STOICH + 1), repeat=self.N_TARGETS):
            if sum(combination) <= self.ORDER * self.MAX_STOICH:
                products_combinations.append(combination)
        products_combinations = np.array(products_combinations)

        combined_combinations = []
        count_by_order = np.full(self.ORDER + 1, 0)
        for ri, r_vals in enumerate(reactants_combinations):
            r_sum = sum(r_vals)
            for pi, p_vals in enumerate(products_combinations):
                p_sum = sum(p_vals)

                # Skip invalid reactions based on the criteria
                bias_cut_off = 2 if self.include_bias else 0

                # Source condition
                if np.sum(r_vals) == 0 and np.sum(p_vals) > bias_cut_off:
                    continue
                # Sink condition
                if np.sum(r_vals) > bias_cut_off and np.sum(p_vals) == 0:
                    continue
                # Max stoichiometry ratio condition
                if np.sum(r_vals) != 0 and np.sum(p_vals) != 0 and np.sum(r_vals) < np.sum(p_vals) / self.MAX_STOICH:
                    continue
                # Redundant reaction condition
                if np.array_equal(r_vals, p_vals):
                    continue

                combined_combinations.append(np.concatenate((r_vals, p_vals)))

                # Count the number of reactions by order
                if p_sum == 0:
                    count_by_order[0] += 1
                else:
                    count_by_order[r_sum] += 1

        # Store reactions with their probabilities
        reaction_with_prob: np.ndarray = np.empty(len(combined_combinations), dtype=object)
        for i, rxn in enumerate(combined_combinations):
            prob = 1 / len(combined_combinations)  # Equal weighting
            reaction_with_prob[i] = (rxn, prob)

        # Check probability sum
        total_prob_sum = sum(rxn[1] for rxn in reaction_with_prob)
        if abs(total_prob_sum - 1.0) > self.TOL:
            raise ValueError(f"Error: sum of probabilities is not 1, got {total_prob_sum}")

        return reaction_with_prob

    def _initialize_population(
        self,
        reaction_list: np.ndarray
    ) -> List[List[np.ndarray]]:
        """
        Initialize the first generation of mechanisms (population) for the genetic algorithm.

        Parameters
        ----------
        reaction_list : np.ndarray
            Array of possible reactions (each as a numpy array).

        Returns
        -------
        population : List[List[np.ndarray]]
            List of mechanisms, where each mechanism is a list of reaction arrays.
        """
        population = []
        i = 0
        while i < self.num_mechs:
            mech_i = []
            species_included = np.full(self.n_targets, False)
            rxn_count = 0
            number_of_rxns = random.randint(self.min_rxns, self.max_rxns)

            while not (np.all(species_included) and rxn_count == number_of_rxns):
                if rxn_count >= self.max_rxns:
                    mech_i = []
                    rxn_count = 0
                    species_included = np.full(self.n_targets, False)

                num_of_rxns = len(reaction_list)
                random_index = random.choice(np.arange(num_of_rxns))
                reaction_i = reaction_list[random_index]

                if array_not_in_list(reaction_i, mech_i):
                    mech_i.append(reaction_i)
                    species_included = species_in_rxn(species_included, reaction_i, self.n_targets)
                    rxn_count += 1

            is_mech_in_list = mech_check(population, mech_i)
            if not is_mech_in_list:
                population.append(mech_i)
                i += 1

        return population

    def _evolve_population(
        self,
        mechanism_list: List[List[np.ndarray]],
        Theta: np.ndarray,
        X: np.ndarray,
        reaction_list: np.ndarray,
        out_file_path: str,
        mech_file_path: str
    ) -> Tuple[List[Any], np.ndarray, np.ndarray]:
        """
        Evolve the population of mechanisms using a genetic algorithm.

        Parameters
        ----------
        mechanism_list : List[List[np.ndarray]]
            Initial population of mechanisms.
        Theta : np.ndarray
            Matrix of observed concentrations over time.
        X : np.ndarray
            Independent variable (e.g., time).
        reaction_list : np.ndarray
            Array of possible reactions.
        out_file_path : str
            Path to write rxn_list.txt.
        mech_file_path : str
            Path to write mech_list.txt.

        Returns
        -------
        best_mechs : list
            List of the best mechanisms after evolution.
        ks : np.ndarray
            Best-fit reaction rate coefficients.
        rm : np.ndarray
            Reaction matrix of the best mechanism.
        """
        out_file = open(out_file_path, "w")
        mech_file = open(mech_file_path, "w")

        best_mechs = []

        # Estimate k values using average logarithmic slope
        k_arr = [np.abs(estimate_k_linearly(X, Theta[:, yi])) for yi in range(Theta.shape[1])]
        k_est = 10 ** np.mean(np.log10(k_arr))

        time = X
        calculated_derivatives = np.gradient(Theta, X, axis=0)
        derivatives_max = np.max(np.abs(calculated_derivatives), axis=0, keepdims=True)
        normalized_calculated_derivatives = calculated_derivatives / derivatives_max
        lowest_error = 10000
        lowest_error_ind = 1
        lowest_error_gen = 1

        for gen_i in range(self.num_gens):
            mech_error = np.full(len(mechanism_list), None)
            for ind, mi in enumerate(mechanism_list):
                reaction_matrix = self._construct_matrix(mi)
                X_coeff, mse = self._fit_coefficients(
                    Theta, reaction_matrix, normalized_calculated_derivatives, derivatives_max, time, k0=k_est
                )
                print(f"Gen {gen_i}, mech {ind}, mse {mse}, lowest_error {lowest_error}, lowest_err_gen {lowest_error_gen}, lowest_error_int {lowest_error_ind}")
                rxn_order = len(X_coeff)
                print(gen_i, ind, mse, rxn_order, file=out_file)
                print(gen_i, ind, reaction_matrix, file=mech_file)
                if mse < lowest_error:
                    lowest_error = mse
                    lowest_error_ind = ind
                    lowest_error_gen = gen_i
                    ks = X_coeff
                    rm = reaction_matrix
                mech_error[ind] = [ind, X_coeff, mse]

            sorted_mech_error = sorted(mech_error, key=lambda x: x[-1])
            n_from_previous_generation = int(self.prev_gen * self.num_mechs)

            prev_gen_mechs = []
            new_best_mechs = []
            for i in range(n_from_previous_generation):
                index = sorted_mech_error[i][0]
                prev_gen_mechs.append(mechanism_list[index])
                new_best_mechs.append([mechanism_list[index], *sorted_mech_error[i]])

            best_mechs = self._update_best_mechs(best_mechs, new_best_mechs, int(2 * self.prev_gen * self.num_mechs))
            print(f"Gen:{gen_i},{len(best_mechs)}")
            print("best mech =", best_mechs[0])

            if gen_i != self.num_gens - 1:
                GA_mechs = generate_new_mechanism(
                    sorted_mech_error,
                    mechanism_list,
                    prev_gen_mechs,
                    self.num_mechs - n_from_previous_generation,
                    selection_pressure=0.2,
                    upper_limit=self.max_rxns,
                    lower_limit=self.min_rxns
                )

                GA_mechs_mutated = generate_mutations(GA_mechs, self.mutation_prob, reaction_list)
                mechanism_list = prev_gen_mechs + GA_mechs_mutated

        out_file.close()
        mech_file.close()
        return best_mechs, ks, rm

    def _construct_matrix(
        self,
        mechanism: List[np.ndarray]
    ) -> np.ndarray:
        """
        Construct a matrix representation of a given reaction mechanism.

        Parameters
        ----------
        mechanism : List[np.ndarray]
            List of reactions, where each reaction is a 1D array: first half for reactants, second half for products.

        Returns
        -------
        Xi : np.ndarray
            Reaction matrix of shape (number of reactions, 2 * n_targets), where each row is [reactants | products].

        Notes
        -----
        - Each row of Xi contains the reactants followed by products for a single reaction.
        - Assumes each reaction array is of length 2 * n_targets.
        """
        Xi = np.zeros((len(mechanism), 2 * self.n_targets), dtype=int)
        for ri, rxn in enumerate(mechanism):
            reactants = rxn[:self.n_targets]
            products = rxn[self.n_targets:]
            Xi[ri] = np.concatenate((reactants, products))
        return Xi

    def _fit_coefficients(self,
        Theta: np.ndarray,
        reaction_matrix: np.ndarray,
        Theta_dot: np.ndarray,
        Theta_dot_max: np.ndarray,
        time: np.ndarray,
        k0=None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit reaction rate coefficients by minimizing the difference between calculated and predicted derivatives.

        Parameters:
        - Theta (array-like): Matrix of reaction data, where each row represents a reaction and
            columns correspond to reactants and products.
        - reaction_matrix (array-like): Matrix representing the reactions for which to fit the coefficients.
        - k0 (None, scalar, or array-like, optional): Initial guesses for the reaction rate constants.
            If None, defaults to ones. If scalar, used for all reactions. If array-like, should match the number of reactions.
        - Theta_dot (array-like): Matrix of the derivative of the reaction data, where each row represents a reaction and
            columns correspond to reactants and products.
        - Theta_dot_max (array-like): list of maximums for each species in Theta_dot
        - time = time array

        Returns:
        - k_fit (array): Fitted reaction rate constants.
        - mse (float): Mean squared error of the fit.

        Raises:
        - ValueError: If `k0` is not None, a scalar, or an array-like object with a length matching the number of reactions.

        Notes:
        - The method calculates the predicted derivatives based on the provided reaction matrix
            and fits the reaction rate constants using a least squares approach.
        - The fitting process minimizes the mean squared error between the calculated and predicted derivatives.
        """

        # Set default initial guesses for k if not provided
        if k0 is None:
            k0 = np.ones(reaction_matrix.shape[0])
        elif np.isscalar(k0):
            k0 = np.array([k0] * reaction_matrix.shape[0])
        elif not isinstance(k0, (np.ndarray, list, tuple)):
            raise ValueError("k0 must be None, a scalar, or an array-like object (e.g., NumPy array, list, tuple).")
        elif len(k0) != reaction_matrix.shape[0]:
            raise ValueError("If k0 is array-like, it must have as many elements as there are reactions in reaction_matrix.")

        # Define the objective function for least squares fitting
        def __objective(k, X, y):
            predicted_derivatives = __predict_derivative(k, X, y)
            normalized_predicted_derivatives = predicted_derivatives / Theta_dot_max
            difference = Theta_dot - normalized_predicted_derivatives
            diff_sum = np.sum(np.abs(difference), axis=1)
            mse = np.mean(np.square(difference))
            return diff_sum, mse


        def __objective_complexity(X):
            mech = X
            complexity = 0
            for rxn in mech:
                complexity += np.sum(rxn)


            return complexity

        # Define the function to predict derivatives based on current coefficients
        def __predict_derivative(k, X, y):
            pred_der = np.zeros_like(y)
            for t_k in range(y.shape[0]):
                #print("t_k =", t_k)
                for r_i, reaction in enumerate(X):
                    reacts = reaction[:reaction.shape[0] // 2]
                    r_i_conc_factor = reactant_concentration(reacts, y[t_k])
                    for s_j in range(len(reacts)):
                        pred_der[t_k][s_j] += (-np.abs(X[r_i][s_j]) * k[r_i] * r_i_conc_factor +
                                            X[r_i][s_j + len(reacts)] * k[r_i] * r_i_conc_factor)
            return pred_der

        # Perform least squares fitting using the `least_squares` function
        result = least_squares(lambda k, X, y: __objective(k, X, y)[0], k0, args=(reaction_matrix, Theta), loss='linear', bounds=(0, np.inf),method="trf", max_nfev = 10)#5bounds=(0, np.inf), method="trf")
        k_fit = result.x
        residuals, mse = __objective(k_fit, reaction_matrix, Theta)
        alpha_complexity = 0.0 # complexity penalty
        mse += alpha_complexity*__objective_complexity(reaction_matrix)
        return k_fit, mse

    def _update_best_mechs(self,
        best_mechs: List[Any],
        new_mechs: List[Any],
        n_best_rxns: int
    ) -> List[Any]:
        """
        Update the list of best mechanisms by combining old and new mechanisms,removing duplicates,
            and sorting by performance.

        Parameters:
        - best_mechs (list of tuples): List of currently best mechanisms, where each mechanism is represented as a tuple.
        - new_mechs (list of tuples): List of newly generated mechanisms to be considered.
        - n_best_rxns (int): Number of top mechanisms to retain after updating.

        Returns:
        - list of tuples: Updated list of best mechanisms, limited to the top `n_best_rxns` based on their performance.

        Notes:
        - Mechanisms are considered duplicates if they contain the same set of reactions, regardless of order.
        - Mechanisms are sorted by their performance metric (e.g., mean squared error) and only the best ones are kept.
        """

        # Combine old and new mechanisms into a single list
        all_mechs = best_mechs + new_mechs

        unique_mechs = []
        seen_mechs = set()
        reaction_set = []

        for mech in all_mechs:
            rxn_matrix = mech[0]
            rxn_ids = []

            # Convert each reaction into a frozenset to account for order-agnostic comparison
            for rxn in rxn_matrix:
                rxn_tuple = tuple(rxn.tolist())  # Convert to list first, then to tuple to avoid numpy-specific behavior
                if rxn_tuple in reaction_set:
                    # Get the index of the matching reaction in the set
                    index = reaction_set.index(rxn_tuple)
                    rxn_ids.append(index)
                else:
                    # Add the reaction to reaction_set and seen_mechs
                    reaction_set.append(rxn_tuple)
                    rxn_ids.append(len(reaction_set) - 1)

            # Check if a mechanism with the same reaction IDs exists
            rxn_ids_tuple = tuple(rxn_ids)
            if rxn_ids_tuple not in seen_mechs:
                seen_mechs.add(rxn_ids_tuple)
                unique_mechs.append(mech)

        # Sort the unique mechanisms by their performance metric (e.g., mean squared error)
        sorted_mechs = sorted(unique_mechs, key=lambda x: x[-1])

        # Return only the top `n_best_rxns` mechanisms
        return sorted_mechs[:n_best_rxns]

def _apply_threshold(
    self,
    ks: np.ndarray,
    reaction_matrix: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply a threshold to filter out reactions whose rate constants (ks) differ by more than MAX_RATIO orders of magnitude.

    Parameters
    ----------
    ks : np.ndarray
        Array of reaction rate constants.
    reaction_matrix : np.ndarray
        Matrix where each row corresponds to the reactants and products of a reaction.

    Returns
    -------
    filtered_ks : np.ndarray
        Rate constants after thresholding.
    filtered_reaction_matrix : np.ndarray
        Reaction matrix after thresholding.
    """

    # Take the base-10 logarithm of the rate constants
    log_ks = np.log10(ks)
    ks_max = np.max(log_ks)
    ks_min = np.min(log_ks)

    # If all log(ks) are within MAX_RATIO, return as is
    if ks_max - ks_min <= self.max_ratio:
        return ks, reaction_matrix
    else:
        # Find indices where log(ks) is within MAX_RATIO of the maximum
        valid_indices = np.where((log_ks <= ks_max) & (log_ks >= ks_max - self.max_ratio))[0]
        # Filter ks and reaction_matrix by valid indices
        filtered_ks = ks[valid_indices]
        filtered_reaction_matrix = np.array(reaction_matrix)[valid_indices]
        return filtered_ks, filtered_reaction_matrix

    def fit(self,
        Theta: np.ndarray,
        X: np.ndarray
    ) -> Tuple['ReactionMechanismGenerator', List[Any]]:
        """
        Fit reaction mechanisms to the provided data by generating possible reactions,
        fitting coefficients, and plotting the results.

        Parameters:
        - Theta (np.ndarray): Matrix of shape (n_samples, n_targets) representing the data.
        - X (np.ndarray): Matrix representing the features for the reactions.

        Returns:
        - self: The instance of the class with fitted mechanisms.
        """
        # Number of target species
        self.N_TARGETS = Theta.shape[1]
        print("number of species  = ", self.N_TARGETS)

        # 1. Generate all possible reactions
        reaction_list = self._generate_reactions()
        print("number of rxns = ", len(reaction_list))

        # 2. Initialize population
        population = self._initialize_population(reaction_list)

        # 3. Evolve population
        best_mechs, ks, rm = self._evolve_population(
            population,
            Theta,
            X,
            reaction_list,
            out_file_path=os.path.abspath(pathname)+"/rxn_list.txt",
            mech_file_path=os.path.abspath(pathname)+"/mech_list.txt"
        )

        return self, mechanism_list


def species_in_rxn(
    species_included: np.ndarray,
    reaction: np.ndarray,
    N_TARGETS: int
) -> np.ndarray:
    """
    Update the list of included species based on whether they appear in a given reaction.

    Parameters:
    - species_included (array-like of bool): Array indicating which species are currently included.
    - reaction (array-like): Array representing a reaction, where the first half corresponds to reactants
      and the second half to products.
    - N_TARGETS (int): Number of target species, used to split the reaction array into reactants and products.

    Returns:
    - array-like of bool: Updated array indicating which species are involved in the reaction.

    Notes:
    - The function marks species as included if they have a non-zero coefficient in either the reactants
      or products of the reaction.
    """

    # Create a copy of the species_included array to update
    species_included_updated = np.copy(species_included)

    # Iterate over each species and check if it is involved in the reaction
    for si in range(species_included.shape[0]):
        if reaction[si] != 0 or reaction[si + N_TARGETS] != 0:
            species_included_updated[si] = True

    return species_included_updated

def reactant_concentration(
    reactants: np.ndarray,
    X: np.ndarray
) -> float:
    """
    Calculate the effective concentration of a set of reactants based on their orders and the current concentrations.

    Parameters:
    - reactants (array-like): List or array of reactant orders where each element corresponds to the
      stoichiometric coefficient of a reactant in the reaction.
    - X (array-like): Current concentrations of all species.

    Returns:
    - float: The effective concentration of the reactants, computed as the product of each reactant's
      concentration raised to the power of its order.

    Notes:
    - The function assumes that reactants are represented by their stoichiometric coefficients.
    - If the order of a reactant is zero, it contributes 1 to the product, effectively ignoring it in the calculation.
    """

    # Initialize the concentration value to 1.0
    value = 1.0

    # Calculate the product of concentrations raised to the power of their corresponding orders
    for i, order in enumerate(reactants):
        if order != 0:
            value *= X[i]**np.abs(order)

    return value

def reaction_system(
    t,
    y: np.ndarray,
    reaction_matrix: List[List[int]],
    ks: np.ndarray
) -> np.ndarray:
    """
    Calculate the rate of change (derivatives) of concentrations for a chemical reaction system.

    Parameters:
    - t (float): Current time (not used in the function but required by ODE solvers).
    - y (array-like): Current concentrations of the species.
    - reaction_matrix (list of lists): Reaction matrix, with the first half as reactants
      and the second half as products.
    - ks (array-like): Array of reaction rate constants, one for each reaction.

    Returns:
    - array-like: Array of derivatives representing the rate of change of concentrations for each species.

    Notes:
    - The function calculates the rate of change of each species based on the given reactions and rate constants.
    - It assumes that each reaction list is split into reactants and products, and updates the derivatives
      according to the reaction stoichiometry and rate constants.
    """

    # Initialize an array to store the predicted derivatives
    pred_derivatives = np.zeros_like(y)

    # Iterate over each reaction
    for ri, reaction in enumerate(reaction_matrix):
        # Extract reactants and products
        reactants = reaction[:len(reaction)//2]

        # Compute the concentration factor for the reactants
        ri_conc_factor = reactant_concentration(reactants, y)

        # Update the derivatives based on reactants and products
        for s_j in range(len(reactants)):
            pred_derivatives[s_j] += (-np.abs(reaction[s_j]) * ks[ri] * ri_conc_factor +
                                      reaction[s_j + len(reactants)] * ks[ri] * ri_conc_factor)

    return pred_derivatives

def mechanism_to_tuple(mech_data):
    mech = mech_data[0]
    # Convert mechanism to a tuple of sorted tuples for consistent comparison
    return tuple(sorted(tuple(sorted(rxns)) for rxns in mech))

def estimate_k_linearly(x, y):
    """
    Estimate the linear rate constant `k` by approximating the slope of the line connecting
    the first data point and the point where the maximum value of `y` occurs.

    Parameters:
    - x (array-like): Independent variable values.
    - y (array-like): Dependent variable values.

    Returns:
    - float: The estimated linear rate constant `k`.

    Notes:
    - If the maximum value of `y` occurs at the first index, the function uses the minimum `y` value
      instead to calculate the slope.
    - The estimated `k` is calculated as the slope of the line connecting the first point
      (x[0], y[0]) and the point with the maximum `y`.
    """

    # Find the index of the maximum value in y
    target_idx = np.argmax(y)

    # If the maximum value is at the first index, use the minimum y value instead
    if target_idx == 0:
        target_idx = np.argmin(y)

    # Extract the maximum y value and its corresponding x value
    max_y = y[target_idx]
    max_x = x[target_idx]

    # Calculate and return the slope between the first point and the point with the maximum y
    return (max_y - y[0]) / (max_x - x[0])

def array_not_in_list(
    arr: np.ndarray,
    list_of_arrays: List[np.ndarray]
) -> bool:
  """Checks if an array is not present in a list of arrays."""

  for other_arr in list_of_arrays:
      if (arr == other_arr).all():
          return False
  return True

def mech_check(
    mechanism_list: List[List[np.ndarray]],
    mech_i: List[np.ndarray]
) -> bool:
  """Checks if mech is already present in a list of arrays."""

  are_equal = False
  is_in_mech_list = False
  for other_mech in mechanism_list:
    permutations = list(itertools.permutations(other_mech, len(other_mech)))
    for other_mech_permutation in permutations:
      are_equal = all(np.array_equal(arr1, arr2) for arr1, arr2 in zip(mech_i, other_mech_permutation))
      if are_equal == True:
        print(are_equal, mech_i, other_mech, other_mech_permutation)
        #print("permutations =", permutations)
        is_in_mech_list = True
        continue #sys.exit()

  return is_in_mech_list
