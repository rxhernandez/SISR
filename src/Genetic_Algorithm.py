import sys
import numpy as np
import itertools
from typing import List, Tuple, Any

def crossover(
    parent1: np.ndarray,
    parent2: np.ndarray,
    UPPER_LIMIT: int,
    LOWER_LIMIT: int
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Performs crossover operation as part of the genetic algorithm.

    Parameters:
        - parent1 (np.ndarray): Reaction matrix of first parent network (N x 2 matrix)
        - parent2 (np.ndarray): Reaction matrix of second parent network (N x 2 matrix)
        - UPPER_LIMIT (int): Maximum reactions per offspring network
        - LOWER_LIMIT (int): Minimum reactions per offspring network

    Returns:
        Tuple[List[np.ndarray], List[np.ndarray]]: Two valid offspring reaction networks
    """

    # Combine parent reaction pools and remove duplicate reactions
    gene_pool = np.concatenate((parent1, parent2))
    unique_reactions = np.unique(gene_pool, axis=0)

    # Enforce maximum network complexity constraints
    viable_reactions = unique_reactions if unique_reactions.shape[0] < gene_pool.shape[0] else gene_pool
    total_available_reactions = viable_reactions.shape[0]
    max_reactions = min(UPPER_LIMIT, total_available_reactions)

    # Initialize species inclusion tracking
    num_species = int(len(gene_pool[0])/2)  # Derived from stoichiometric matrix dimensions
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
        if (_validate_network(offspring1, num_species) and
            _validate_network(offspring2, num_species)):
            return offspring1, offspring2

        attempts += 1

    raise RuntimeError(f"Failed to generate valid mechanisms after {MAX_ATTEMPTS} attempts")

def _validate_network(
    network: List[np.ndarray],
    num_species: int
) -> bool:
    """Validates that all species participate in at least one reaction"""
    species_present = np.zeros(num_species, dtype=bool)
    for reaction in network:
        # Split reaction into reactants and products
        reactants = reaction[:num_species].astype(bool)
        products = reaction[num_species:].astype(bool)

        # Combine using logical OR to find participating species
        participating = reactants | products

        # Update presence tracking
        species_present |= participating

    return bool(np.all(species_present))

def generate_new_mechanism(
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
        mech_ids_sorted (List[Any]): Sorted indices or identifiers for mechanisms by fitness.
        mech_list (List[List[np.ndarray]]): List of mechanisms, each a list of reactions.
        prev_gen_mechs (List[List[np.ndarray]]): Mechanisms from the previous generation.
        NUM_NEW_MECHS (int): Number of new mechanisms to generate.
        selection_pressure (float): Selection pressure parameter for parent selection probabilities.
        UPPER_LIMIT (int): Maximum number of reactions per mechanism.
        LOWER_LIMIT (int): Minimum number of reactions per mechanism.

    Returns:
        List[List[np.ndarray]]: List of newly generated mechanisms.
    """

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
        parent_indices = np.random.choice(
            range(num_mechs), size=2, replace=False, p=probabilities
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
            print(
                "Overlap detected in crossover: finished generating",
                len(next_gen_mechs), "of", NUM_NEW_MECHS
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
        mech_list (List[List[np.ndarray]]): List of mechanisms, each a list of reactions.
        mut_prob (float): Mutation probability per mechanism.
        reactions_list (List[np.ndarray]): Pool of all possible reactions.
        MAX_ATTEMPTS (int): Maximum attempts to find a valid replacement reaction.
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
            print(f"Warning: Could not find valid mutation for mechanism {mech_idx}, reaction {reaction_idx} after {MAX_ATTEMPTS} attempts.")

def array_not_in_list(
    arr: np.ndarray,
    list_of_arrays: List[np.ndarray]
) -> bool:
    """
    Checks if an array is not present in a list of arrays.
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
        species_included (np.ndarray): Boolean array indicating which species are currently included.
        reaction (np.ndarray): Array representing a reaction (first half: reactants, second half: products).
        N_TARGETS (int): Number of species.

    Returns:
        np.ndarray: Updated boolean array indicating which species are involved in the mechanism.
    """
    updated = np.copy(species_included)
    for si in range(N_TARGETS):
        if reaction[si] != 0 or reaction[si + N_TARGETS] != 0:
            updated[si] = True
    return updated

def mech_check(
    mechanism_list: List[List[np.ndarray]],
    mech_i: List[np.ndarray]
) -> bool:
    """
    Checks if a mechanism is already present in a list of mechanisms,
    considering all possible reaction orders.

    Parameters:
        mechanism_list (List[List[np.ndarray]]): List of mechanisms (each a list of reactions).
        mech_i (List[np.ndarray]): The mechanism to check.

    Returns:
        bool: True if mech_i is present in mechanism_list (up to reaction order), else False.
    """
    for other_mech in mechanism_list:
        if len(other_mech) != len(mech_i):
            continue  # Mechanisms of different lengths cannot be equal
        # Check all permutations of other_mech for equality with mech_i
        for permuted_mech in itertools.permutations(other_mech):
            if all(np.array_equal(r1, r2) for r1, r2 in zip(mech_i, permuted_mech)):
                return True
    return False

    # --- Example Usage ---
if __name__ == "__main__":

    # 1. Define sample reactions
    # Reactions are represented as NumPy arrays. First half are reactants, second half products.
    # E.g., [1, 0, 0, 0, 1, 0, 0, 0] represents A -> B (assuming A is species 0, B is species 1).
    reaction_A_to_B = np.array([1, 0, 0, 0, 1, 0, 0, 0], dtype=int)  # A -> B
    reaction_B_to_C = np.array([0, 1, 0, 0, 0, 1, 0, 0], dtype=int)  # B -> C
    reaction_C_to_A = np.array([0, 0, 1, 0, 1, 0, 0, 0], dtype=int)  # C -> A
    reaction_AB_to_C = np.array([1, 1, 0, 0, 0, 0, 1, 0], dtype=int)  # A + B -> C

    # 2. Create a reaction list
    reactions_list = [reaction_A_to_B, reaction_B_to_C, reaction_C_to_A, reaction_AB_to_C]

    # 3. Create initial mechanisms
    # Each mechanism is a list of reaction arrays.
    initial_mechanism1 = [reaction_A_to_B, reaction_B_to_C]
    initial_mechanism2 = [reaction_C_to_A, reaction_AB_to_C]
    mech_list = [initial_mechanism1, initial_mechanism2]

    # 4. Simulate sorted mechanism IDs (replace with your actual fitness sorting)
    mech_ids_sorted = [(0,), (1,)]  # Indices representing sorted mechanisms

    # 5. Set genetic algorithm parameters
    NUM_NEW_MECHS = 2
    selection_pressure = 0.2
    UPPER_LIMIT = 4  # Max reactions per mechanism
    LOWER_LIMIT = 1  # Min reactions per mechanism

    # 6. Generate new mechanisms
    prev_gen_mechs = mech_list  # The current list becomes the previous generation for demonstration
    new_mechs = generate_new_mechanism(
        mech_ids_sorted, mech_list, prev_gen_mechs, NUM_NEW_MECHS,
        selection_pressure, UPPER_LIMIT, LOWER_LIMIT
    )

    print("Original mechanisms:", mech_list)
    print("New mechanisms:", new_mechs)

    # 7. Apply mutations
    mutation_probability = 0.1
    generate_mutations(mech_list, mutation_probability, reactions_list)  # mech_list is modified in place

    print("Mechanisms after mutation:", mech_list)
