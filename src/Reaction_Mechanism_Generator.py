import os
import sys
import random
import logging
import itertools
import numpy as np
from math import comb
import time as real_time
import scipy.constants as sp
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from numpy.core.numeric import isscalar
from scipy.optimize import least_squares
from typing import List, Tuple, Any

# Functions for genetic algorithm
from .genetic_algorithm import mech_check
from .genetic_algorithm import generate_mutations
from .genetic_algorithm import species_in_reaction
from .genetic_algorithm import next_generation_mechanism

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
        verbosity=1
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
        self.PREV_GEN = from_previous_generation
        self.NUM_GENS = num_generations
        self.TOL = tol
        self.MAX_RATIO = max_ratio # max ratio of largest coeff / threshold
        self.MAX_STOICH = max_stoichiometric_ratio
        self.NUM_MECHS = num_mech_per_gen
        self.MAX_RXN = max_rxns_per_mech
        self.MIN_RXN = min_rxns_per_mech
        self.P_MUTATION = mutation_prob
        self.N_TARGETS = 0
        self.reaction_mechanism: List[np.ndarray] = []
        self.rate_constants: np.ndarray = np.array([])
        self.mse = 0

        # Setup logging
        setup_logging(verbosity)

    def _generate_reactions(self) -> List[np.ndarray]:
        """
        Generate all possible reactions for a given number of target species
        with associated probabilities.

        Returns
        - reaction_list (array-like): An array which contains reactions (as a numpy array).
        """
        # if N_targets is set to 0, return an empty list
        if self.N_TARGETS == 0:
            return []

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
        reaction_list: List[np.ndarray] = []
        for i, rxn in enumerate(combined_combinations):
            reaction_list.append(np.array(rxn))

        return reaction_list


    def _initialize_population(
        self,
        reaction_list: List[np.ndarray]
    ) -> List[List[np.ndarray]]:
        """
        Initialize the first generation of mechanisms (population) for the genetic algorithm.

        Parameters
        - reaction_list (array-like): Array of possible reactions (each as a numpy array).

        Returns
        - population (array-like): List of mechanisms, where each mechanism
        is a list of reaction arrays.
        """
        population: List[List[np.ndarray]] = []
        mech_count = 0

        MAX_ATTEMPTS = self.NUM_MECHS*100  # Prevent infinite loops in pathological cases
        attempts = 0

        while attempts < MAX_ATTEMPTS:
            attempts += 1
            mech_i: List[np.ndarray] = []
            species_included = np.full(self.N_TARGETS, False)
            rxn_count = 0
            number_of_rxns = random.randint(self.MIN_RXN, self.MAX_RXN)

            while not (np.all(species_included) and rxn_count == number_of_rxns):
                if rxn_count >= self.MAX_RXN:
                    mech_i = []
                    rxn_count = 0
                    species_included = np.full(self.N_TARGETS, False)

                num_of_rxns = len(reaction_list)
                random_index = random.choice(np.arange(num_of_rxns))
                reaction_i = reaction_list[random_index]

                if array_not_in_list(reaction_i, mech_i):
                    mech_i.append(reaction_i)
                    species_included = species_in_reaction(species_included, reaction_i, self.N_TARGETS)
                    rxn_count += 1

            is_mech_in_pop = mech_check(population, mech_i)
            if not is_mech_in_pop:
                population.append(mech_i)
                mech_count += 1

            # if we have enough mechanisms, break out of the loop
            if mech_count >= self.NUM_MECHS:
                break

        return population


    def _evolve_population(
        self,
        mechanism_list: List[List[np.ndarray]],
        S: np.ndarray,
        X: np.ndarray,
        reaction_list: List[np.ndarray],
        out_file_path: str,
        mech_file_path: str
    ) -> Tuple[List[Any], np.ndarray, List[np.ndarray], float]:
        """
        Evolve the population of mechanisms using a genetic algorithm.

        Parameters
        - mechanism_list (array-like): Initial population of mechanisms.
        - S (array-like): Matrix of observed concentrations over time.
        - X (array-like): Independent variable (e.g., time).
        - reaction_list (array-like): Array of possible reactions.
        - out_file_path (str): Path to write rxn_list.txt.
        - mech_file_path (str): Path to write mech_list.txt.

        Returns
        - best_mechs (array-like): List of the best mechanisms after evolution.
        - rate_consts (array-like): Best-fit reaction rate coefficients.
        - reaction_mech (array-like): Best mechanism.
        """
        out_file = open(out_file_path, "w")
        mech_file = open(mech_file_path, "w")

        # Initialize variables
        best_mechs: List[Any] = []
        rate_consts: np.ndarray = np.array([])
        reaction_mech: List[np.ndarray] = []
        best_mse = 0

        # Estimate k values using average logarithmic slope
        k_arr = [np.abs(estimate_k_linearly(X, np.array(S)[:, yi])) for yi in range(np.array(S).shape[1])]
        k_est = 10 ** np.mean(np.log10(k_arr))

        time = X
        calculated_derivatives = np.gradient(S, X, axis=0)
        derivatives_max = np.max(np.abs(calculated_derivatives), axis=0, keepdims=True)
        normalized_calculated_derivatives = calculated_derivatives / derivatives_max
        lowest_error = 10000
        lowest_error_ind = 1
        lowest_error_gen = 1

        for gen_i in range(self.NUM_GENS):
            start = real_time.time()
            mech_error = np.full(len(mechanism_list), None)
            for ind, mi in enumerate(mechanism_list):
                reaction_mechanism = self._construct_matrix(mi)
                rate_coeff, mse = self._fit_coefficients(
                    S, reaction_mechanism, normalized_calculated_derivatives, derivatives_max, time, k0=k_est
                )
                logging.debug(f"Gen {gen_i}, mech {ind}, mse {mse}, lowest_error {lowest_error}, lowest_err_gen {lowest_error_gen}, lowest_error_int {lowest_error_ind}")
                rxn_order = len(rate_coeff)
                print(gen_i, ind, mse, rxn_order, file=out_file)
                print(gen_i, ind, reaction_mechanism, file=mech_file)
                if mse < lowest_error:
                    lowest_error = mse
                    lowest_error_ind = ind
                    lowest_error_gen = gen_i
                    rate_consts = rate_coeff
                    reaction_mech = reaction_mechanism
                    best_mse = mse
                mech_error[ind] = [ind, rate_coeff, mse]

            sorted_mech_error = sorted(mech_error, key=lambda x: x[-1])
            n_from_previous_generation = int(self.PREV_GEN * self.NUM_MECHS)

            prev_gen_mechs = []
            new_best_mechs = []
            for i in range(n_from_previous_generation):
                index = sorted_mech_error[i][0]
                prev_gen_mechs.append(mechanism_list[index])
                new_best_mechs.append([mechanism_list[index], *sorted_mech_error[i]])

            self._update_best_mechs(best_mechs, new_best_mechs, int(2 * self.PREV_GEN * self.NUM_MECHS))

            if gen_i != self.NUM_GENS - 1:
                GA_mechs = next_generation_mechanism(
                    sorted_mech_error,
                    mechanism_list,
                    prev_gen_mechs,
                    self.NUM_MECHS - n_from_previous_generation,
                    selection_pressure=0.2,
                    UPPER_LIMIT=self.MAX_RXN,
                    LOWER_LIMIT=self.MIN_RXN
                )

                # Carry out mutations on the new mechanisms
                generate_mutations(GA_mechs, self.P_MUTATION, reaction_list)
                mechanism_list = prev_gen_mechs + GA_mechs

            # Output enlapsed time
            elapsed = real_time.time() - start
            hours, rem = divmod(int(elapsed), 3600)
            minutes, seconds = divmod(rem, 60)
            logging.info(f"Generation {(gen_i+1):d}/{self.NUM_GENS:d} completed in {hours:02d}:{minutes:02d}:{seconds:02d}")
            logging.debug(f"Best mechanism: {best_mechs[0][0]} with MSE {best_mechs[0][2]}")

        out_file.close()
        mech_file.close()
        return best_mechs, rate_consts, reaction_mech, best_mse


    def _construct_matrix(
        self,
        mechanism: List[np.ndarray]
    ) -> List[np.ndarray]:
        """
        Construct a matrix representation of a given reaction mechanism.

        Parameters
        - mechanism (array-like): List of reactions, where each reaction is a 1D array: first half for reactants, second half for products.

        Returns
        - Xi (array-like): Reaction matrix of shape (number of reactions, 2 * N_TARGETS), where each row is [reactants | products].

        Notes
        - Each row of Xi contains the reactants followed by products for a single reaction.
        - Assumes each reaction array is of length 2 * N_TARGETS.
        """

        Xi: List[np.ndarray] = np.empty((len(mechanism), 2 * self.N_TARGETS), dtype=int).tolist()
        for ri, rxn in enumerate(mechanism):
            reactants = rxn[:self.N_TARGETS]
            products = rxn[self.N_TARGETS:]
            Xi[ri] = np.concatenate((reactants, products))
        return Xi


    def _fit_coefficients(self,
        S: np.ndarray,
        reaction_mechanism: List[np.ndarray],
        S_dot: List[np.ndarray],
        S_dot_max: np.ndarray,
        time: np.ndarray,
        k0=None
    ) -> Tuple[np.ndarray, float]:
        """
        Fit reaction rate coefficients by minimizing the difference between calculated and predicted derivatives.

        Parameters:
        - S (array-like): Matrix of reaction data, where each row represents a reaction and
            columns correspond to reactants and products.
        - reaction_mechanism (array-like): Matrix representing the reactions for which to fit the coefficients.
        - k0 (None, scalar, or array-like, optional): Initial guesses for the reaction rate constants.
            If None, defaults to ones. If scalar, used for all reactions. If array-like, should match the number of reactions.
        - S_dot (array-like): Matrix of the derivative of the reaction data, where each row represents a reaction and
            columns correspond to reactants and products.
        - S_dot_max (array-like): list of maximums for each species in S_dot
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
            k0 = np.ones(len(reaction_mechanism))
        elif np.isscalar(k0):
            k0 = np.array([k0] * len(reaction_mechanism))
        elif not isinstance(k0, (np.ndarray, list, tuple)):
            raise ValueError("k0 must be None, a scalar, or an array-like object (e.g., NumPy array, list, tuple).")
        elif len(k0) != len(reaction_mechanism):
            raise ValueError("If k0 is array-like, it must have as many elements as there are reactions in reaction_mechanism.")
        elif S is None or S_dot is None:
            raise ValueError("S and S_dot must be provided for fitting coefficients.")

        # Define the objective function for least squares fitting
        def __objective(k, X, y):
            predicted_derivatives = __predict_derivative(k, X, y)
            normalized_predicted_derivatives = predicted_derivatives / S_dot_max
            difference = S_dot - normalized_predicted_derivatives
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
                for r_i, reaction in enumerate(X):
                    reacts = reaction[:reaction.shape[0] // 2]
                    r_i_conc_factor = reactant_concentration(reacts, y[t_k])
                    for s_j in range(len(reacts)):
                        pred_der[t_k][s_j] += (-np.abs(X[r_i][s_j]) * k[r_i] * r_i_conc_factor +
                                            X[r_i][s_j + len(reacts)] * k[r_i] * r_i_conc_factor)
            return pred_der

        # Perform least squares fitting using the `least_squares` function
        result = least_squares(lambda k, X, y: __objective(k, X, y)[0], k0, args=(reaction_mechanism, S), loss='linear', bounds=(0, np.inf),method="trf", max_nfev = 10)#5bounds=(0, np.inf), method="trf")
        k_fit = result.x
        residuals, mse = __objective(k_fit, reaction_mechanism, S)
        return k_fit, mse


    def _update_best_mechs(self,
        best_mechs: List[Any],
        new_mechs: List[Any],
        n_best_rxns: int
    ) -> None:
        """
        Update the list of best mechanisms by combining old and new mechanisms,removing duplicates,
            and sorting by performance.

        Parameters:
        - best_mechs (array-like): List of currently best mechanisms, where each mechanism is represented as a tuple.
        - new_mechs (array-like): List of newly generated mechanisms to be considered.
        - n_best_rxns (int): Number of top mechanisms to retain after updating.

        Returns:
        - None: Updated list of best mechanisms, limited to the top `n_best_rxns` based on their performance.

        Notes:
        - Mechanisms are considered duplicates if they contain the same set of reactions, regardless of order.
        - Mechanisms are sorted by their performance metric (e.g., mean squared error) and only the best ones are kept.
        """

        if n_best_rxns <= 0:
            best_mechs.clear()
            return

        # Combine all mechanisms and initialize tracking structures
        all_mechs = best_mechs + new_mechs
        unique_mechs = []
        seen_mechs_hashes = set()

        # Implement dictonary to store unique mechanisms based on reaction IDs
        reation_to_id = {}
        next_reaction_id = 0

        for mech in all_mechs:
            # Generate a unique hash for this mechanism's reaction combination
            rxn_ids = []

            # Convert each reaction into a immutable type (tuple) for hashing
            for rxn in mech[0]:
                rxn_tuple = tuple(rxn.tolist()) if hasattr(rxn, 'tolist') else tuple(rxn)

                # Get or assign a unique ID for the reaction
                if rxn_tuple not in reation_to_id:
                    reation_to_id[rxn_tuple] = next_reaction_id
                    next_reaction_id += 1
                rxn_ids.append(reation_to_id[rxn_tuple])

            # Create a hashable representation of the mechanism using reaction IDs
            mech_hash = frozenset(rxn_ids)

            if mech_hash not in seen_mechs_hashes:
                seen_mechs_hashes.add(mech_hash)
                unique_mechs.append(mech)

        # Sort the unique mechanisms by their performance metric: MSE
        unique_mechs.sort(key=lambda x: x[-1])

        # Return only the top `n_best_rxns` mechanisms
        best_mechs.clear()
        best_mechs.extend(unique_mechs[:n_best_rxns])


    def _apply_threshold(
        self,
        ks: np.ndarray,
        reaction_mechanism: List[np.ndarray]
    ) -> Tuple[np.ndarray, List[np.ndarray]]:
        """
        Apply a threshold to filter out reactions whose rate constants (ks) differ by more than MAX_RATIO orders of magnitude.

        Parameters
        - ks (array-like): Array of reaction rate constants.
        - reaction_mechanism (array-like): Matrix where each row corresponds to the reactants and products of a reaction.

        Returns
        - filtered_ks (array-like): Rate constants after thresholding.
        - filtered_reaction_mechanism (array-like): Reaction matrix after thresholding.
        """

        # Take the base-10 logarithm of the rate constants
        log_ks = np.log10(ks)
        ks_max = np.max(log_ks)
        ks_min = np.min(log_ks)

        # If all log(ks) are within MAX_RATIO, return as is
        if ks_max - ks_min <= self.MAX_RATIO:
            return ks, reaction_mechanism
        else:
            # Find indices where log(ks) is within MAX_RATIO of the maximum
            valid_indices = np.where((log_ks <= ks_max) & (log_ks >= ks_max - self.MAX_RATIO))[0]
            # Filter ks and reaction_mechanism by valid indices
            filtered_ks = ks[valid_indices]
            filtered_reaction_mechanism = [reaction_mechanism[i] for i in valid_indices]
            return filtered_ks, filtered_reaction_mechanism


    def fit(self,
        S: np.ndarray,
        X: np.ndarray
    ) -> 'ReactionMechanismGenerator':
        """
        Fit reaction mechanisms to the provided data by generating possible reactions,
        fitting coefficients, and plotting the results.

        Parameters:
        - S (array-like): Matrix of shape (n_samples, N_TARGETS) representing the data.
        - X (array-like): Matrix representing the features for the reactions.

        Returns:
        - self: The instance of the class with fitted mechanisms.
        """
        start = real_time.time()

        # Number of target species
        self.N_TARGETS = np.array(S).shape[1]
        logging.info(f"Number of species  = {self.N_TARGETS}")

        # 1. Generate all possible reactions
        reaction_list = self._generate_reactions()
        logging.info(f"Number of reactions = {len(reaction_list):d}")

        # 2. Initialize population
        population = self._initialize_population(reaction_list)

        # 3. Evolve population
        mechanism_list = self._evolve_population(
            population,
            S,
            X,
            reaction_list,
            out_file_path=os.path.abspath(pathname)+"/rxn_list.txt",
            mech_file_path=os.path.abspath(pathname)+"/mech_list.txt"
        )

        elapsed = real_time.time() - start
        hours, rem = divmod(int(elapsed), 3600)
        minutes, seconds = divmod(rem, 60)
        logging.info(f"Time taken for evolution: {hours:02d}:{minutes:02d}:{seconds:02d}")

        # 4. Extract best mechanism and its coefficients
        best_mechs, rate_consts, reaction_mech, best_mse = mechanism_list
        self.reaction_mechanism = reaction_mech
        self.rate_constants = rate_consts
        self.mse = best_mse
        return self


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
    - value (float): The effective concentration of the reactants, computed as the product of each reactant's
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
    t: List[float],
    y: np.ndarray,
    reaction_mechanism: List[np.ndarray],
    ks: np.ndarray
) -> np.ndarray:
    """
    Calculate the rate of change (derivatives) of concentrations for a chemical reaction system.

    Parameters:
    - t (float): Current time (not used in the function but required by ODE solvers).
    - y (array-like): Current concentrations of the species.
    - reaction_mechanism (list of lists): Reaction matrix, with the first half as reactants
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
    for ri, reaction in enumerate(reaction_mechanism):
        # Extract reactants and products
        reactants = reaction[:len(reaction)//2]

        # Compute the concentration factor for the reactants
        ri_conc_factor = reactant_concentration(reactants, y)

        # Update the derivatives based on reactants and products
        for s_j in range(len(reactants)):
            pred_derivatives[s_j] += (-np.abs(reaction[s_j]) * ks[ri] * ri_conc_factor +
                                      reaction[s_j + len(reactants)] * ks[ri] * ri_conc_factor)

    return pred_derivatives


def estimate_k_linearly(x, y):
    """
    Estimate the linear rate constant `k` by approximating the slope of the line connecting
    the first data point and the point where the maximum value of `y` occurs.

    Parameters:
    - x (array-like): Independent variable values.
    - y (array-like): Dependent variable values.

    Returns:
    - (float): The estimated linear rate constant `k`.

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
    """
    Checks if an array is not present in a list of arrays.

    Parameters:
    - arr (array-like): target array to check.
    - list_of_arrays (array-like): list of arrays to check against.

    Returns:
    - (bool): True if the array is not present in the list, False otherwise.
    """

    for other_arr in list_of_arrays:
        if (arr == other_arr).all():
            return False
    return True

def setup_logging(verbosity):
    # Map verbosity (0=WARNING, 1=INFO, 2=DEBUG)
    levels = [logging.WARNING, logging.INFO, logging.DEBUG]
    level = levels[min(verbosity, 2)]
    logging.basicConfig(level=level, format='%(levelname)s - %(message)s')
