import sys
import logging
import itertools
import numpy as np
from typing import List, Tuple, Any

logger = logging.getLogger(__name__)

def crossover(
    parent1: List[np.ndarray],
    parent2: List[np.ndarray],
    UPPER_LIMIT: int,
    LOWER_LIMIT: int
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Performs crossover operation as part of the genetic algorithm.

    Parameters:
    - parent1 (array-like): Reaction matrix of first parent network (N x 2 matrix)
    - parent2 (array-like): Reaction matrix of second parent network (N x 2 matrix)
    - UPPER_LIMIT (int): Maximum reactions per offspring network
    - LOWER_LIMIT (int): Minimum reactions per offspring network

    Returns:
    - Tuple[List[np.ndarray], List[np.ndarray]]: Two valid offspring reaction networks

    Raises:
    - RuntimeError: If valid offspring cannot be generated after a set number of attempts.
    """

    # Combine parent reaction pools and remove duplicate reactions
    reaction_pool = np.concatenate((parent1, parent2))
    unique_reactions = np.unique(reaction_pool, axis=0)

    # Enforce maximum network complexity constraints
    viable_reactions = unique_reactions if unique_reactions.shape[0] < reaction_pool.shape[0] else reaction_pool
    total_available_reactions = viable_reactions.shape[0]
    max_reactions = min(UPPER_LIMIT, total_available_reactions)

    # Initialize species inclusion tracking
    num_species = int(len(reaction_pool[0])/2)  # Derived from stoichiometric matrix dimensions
    MAX_ATTEMPTS = 500  # Prevent infinite loops in pathological cases
    attempts = 0

    while attempts < MAX_ATTEMPTS:
        # Randomly select network sizes within constrained bounds
        offspring1_size = np.random.randint(LOWER_LIMIT, max_reactions + 1)
        offspring2_size = np.random.randint(LOWER_LIMIT, max_reactions + 1)

        # Create candidate networks through random reaction selection
        offspring1_idx = np.random.choice(range(total_available_reactions), offspring1_size, replace=False)
        offspring2_idx = np.random.choice(range(total_available_reactions), offspring2_size, replace=False)

        offspring1 = [viable_reactions[idx] for idx in offspring1_idx]
        offspring2 = [viable_reactions[idx] for idx in offspring2_idx]

        # Verify species participation
        if (validate_mechanism(offspring1, num_species) and
            validate_mechanism(offspring2, num_species)):
            return offspring1, offspring2

        attempts += 1

    raise RuntimeError(f"Failed to generate valid mechanisms after {MAX_ATTEMPTS} attempts")


def validate_mechanism(
    mechanism: List[np.ndarray],
    num_species: int
) -> bool:
    """
    Validates that all species participate in at least one reaction.

    Parameters:
    - network (array-like): List of reactions in the network.
    - num_species (int): Total number of species.

    Returns:
    - bool: True if all species are present in the network, False otherwise.
    """
    species_present = np.zeros(num_species, dtype=bool)
    for reaction in mechanism:
        # Split reaction into reactants and products
        reactants = reaction[:num_species].astype(bool)
        products = reaction[num_species:].astype(bool)

        # Combine using logical OR to find participating species
        participating = reactants | products

        # Update presence tracking
        species_present |= participating

    return bool(np.all(species_present))


def next_generation_mechanism(
    mech_ids_sorted: List[Any],
    mech_list: List[List[np.ndarray]],
    prev_gen_mechs: List[List[np.ndarray]],
    NUM_NEW_MECHS: int,
    selection_pressure: float,
    UPPER_LIMIT: int,
    LOWER_LIMIT: int
) -> List[List[np.ndarray]]:
    """
    Generate a new population of reaction mechanisms using genetic algorithm.

    Parameters:
    - mech_ids_sorted (array-like): Sorted indices or identifiers for mechanisms by fitness.
    - mech_list (array-like): List of mechanisms, each a list of reactions.
    - prev_gen_mechs (array-like): Mechanisms from the previous generation.
    - NUM_NEW_MECHS (int): Number of new mechanisms to generate.
    - selection_pressure (float): Selection pressure parameter for parent selection probabilities.
    - UPPER_LIMIT (int): Maximum number of reactions per mechanism.
    - LOWER_LIMIT (int): Minimum number of reactions per mechanism.

    Returns:
    - next_gen_mechs (array-like): List of newly generated mechanisms.

    Raise:
    - ValueError: If the number of mechanisms in prev_gen_mechs is less than 2.
    """

    if len(prev_gen_mechs) < 2:
        raise ValueError("At least two mechanisms are required to generate a new generation.")

    next_gen_mechs = []
    num_mechs = len(prev_gen_mechs)

    # Compute selection probabilities based on rank (higher rank = higher probability)
    probabilities = np.array([
        (selection_pressure + (num_mechs - i)) for i in range(num_mechs)
    ])
    probabilities = probabilities / np.sum(probabilities)

    # Generate new mechanisms through crossover until desired number is reached
    while len(next_gen_mechs) < NUM_NEW_MECHS:
        # Select two distinct parent mechanisms based on selection probabilities
        rng = np.random.default_rng()
        parent_indices = rng.choice(
            num_mechs, size=2, replace=False, p=probabilities
        )
        parent_id1, parent_id2 = mech_ids_sorted[parent_indices[0]], mech_ids_sorted[parent_indices[1]]
        parent1 = mech_list[parent_id1[0]]
        parent2 = mech_list[parent_id2[0]]

        # Crossover: combine reactions from both parents to create two offspring mechanisms
        offspring1, offspring2 = crossover(parent1, parent2, UPPER_LIMIT, LOWER_LIMIT)

        # Ensure offspring mechanisms are not duplicates of previous generation
        is_off1_new = not mech_check(prev_gen_mechs, offspring1)
        is_off2_new = not mech_check(prev_gen_mechs, offspring2)
        if is_off1_new and is_off2_new:
            next_gen_mechs.extend([offspring1, offspring2])
        else:
            logger.debug(
                f"Overlap detected in crossover: finished generating {len(next_gen_mechs)} of {NUM_NEW_MECHS}"
            )

    # Truncate to the requested number of mechanisms (in case of over-generation)
    next_gen_mechs = next_gen_mechs[:NUM_NEW_MECHS]
    return next_gen_mechs


def generate_mutations(
    mech_list: List[List[np.ndarray]],
    mut_prob: float,
    reactions_list: List[np.ndarray],
    MAX_ATTEMPTS: int = 1000
) -> None:
    """
    Applies mutation operations to a population of mechanisms.
    Each mutation replaces a reaction in a randomly chosen mechanism with a new reaction,
    ensuring all species remain represented and no duplicate reactions are introduced.

    Parameters:
    - mech_list (array-like): List of mechanisms, each a list of reactions.
    - mut_prob (float): Mutation probability per mechanism.
    - reactions_list (array-like): Pool of all possible reactions.
    - MAX_ATTEMPTS (int): Maximum attempts to find a valid replacement reaction.

    Returns:
    - None: Modifies mech_list in place.

    Notes:
    - The mutation process is probabilistic and may not mutate all mechanisms.
    - The function ensures that the new reaction does not duplicate any existing reactions
    in the mechanism and that all species are still represented.
    - The function modifies mech_list in place, so the original list is updated.
    """
    num_mechs = len(mech_list)
    lambda_mut = mut_prob * num_mechs
    num_mutations = int(lambda_mut)

    for _ in range(num_mutations):
        # Randomly select a mechanism and a reaction within it
        mech_idx = np.random.randint(0, num_mechs)
        selected_mech = mech_list[mech_idx]
        reaction_idx = np.random.randint(0, len(selected_mech))
        number_of_species = int(len(selected_mech[reaction_idx]) / 2)

        # Track which species are included by the remaining reactions
        species_included = np.full(number_of_species, False)
        for i, rxn in enumerate(selected_mech):
            if i == reaction_idx:
                continue
            species_included = species_in_reaction(species_included, rxn, number_of_species)

        # Try to find a valid new reaction to replace the selected one
        for attempt in range(MAX_ATTEMPTS):
            new_reaction_idx = np.random.randint(0, len(reactions_list))
            new_reaction = reactions_list[new_reaction_idx]
            if array_not_in_list(new_reaction, selected_mech):
                temp_species_included = species_in_reaction(species_included, new_reaction, number_of_species)
                if np.all(temp_species_included):
                    # Apply the mutation
                    mech_list[mech_idx][reaction_idx] = new_reaction
                    break
        else:
            # If we exit the loop without a break, mutation was not possible
            logger.debug(f"Warning: Could not find valid mutation for mechanism {mech_idx}, reaction {reaction_idx} after {MAX_ATTEMPTS} attempts.")


def array_not_in_list(
    arr: np.ndarray,
    list_of_arrays: List[np.ndarray]
) -> bool:
    """
    Checks if an array is not present in a list of arrays.

    Parameters:
    - arr (array-like): The array to check.
    - list_of_arrays (array-like): The list of arrays to check against.

    Returns:
    - bool: True if arr is not in list_of_arrays, False otherwise.
    """
    for other_arr in list_of_arrays:
        if np.array_equal(arr, other_arr):
            return False
    return True


def species_in_reaction(
    species_included: np.ndarray,
    reaction: np.ndarray,
    N_TARGETS: int
) -> np.ndarray:
    """
    Updates the array of included species based on their presence in the given reaction.

    Parameters:
    - species_included (np.ndarray): Boolean array indicating which species are currently included.
    - reaction (np.ndarray): Array representing a reaction (first half: reactants, second half: products).
    - N_TARGETS (int): Number of species.

    Returns:
    - np.ndarray: Updated boolean array indicating which species are involved in the mechanism.
    Notes:
    - The function marks species as included if they have a non-zero coefficient in either the reactants
        or products of the reaction.
    """
    species_included_updated = np.copy(species_included)
    for si in range(N_TARGETS):
        if reaction[si] != 0 or reaction[si + N_TARGETS] != 0:
            species_included_updated[si] = True
    return species_included_updated


def mech_check(
    mechanism_list: List[List[np.ndarray]],
    mech_i: List[np.ndarray]
) -> bool:
    """
    Checks if a mechanism is already present in a list of mechanisms,
    considering all possible reaction orders.

    Parameters:
    - mechanism_list (List[List[np.ndarray]]): List of mechanisms (each a list of reactions).
    - mech_i (List[np.ndarray]): The mechanism to check.

    Returns:
    - bool: True if mech_i is present in mechanism_list (up to reaction order), else False.
    """
    for other_mech in mechanism_list:
        if len(other_mech) != len(mech_i):
            continue  # Mechanisms of different lengths cannot be equal
        # Check all permutations of other_mech for equality with mech_i
        for permuted_mech in itertools.permutations(other_mech):
            if all(np.array_equal(r1, r2) for r1, r2 in zip(mech_i, permuted_mech)):
                return True
    return False
