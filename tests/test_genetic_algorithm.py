import unittest
import numpy as np
from src.SISR.genetic_algorithm import (
    crossover,
    validate_mechanism,
    next_generation_mechanism,
    generate_mutations,
    array_not_in_list,
    species_in_reaction,
    mech_check
)

class TestGeneticAlgorithm(unittest.TestCase):
    def setUp(self):
        # Define sample reactions for testing
        # Format: First half are reactants, second half are products
        self.reaction_A_to_B = np.array([1, 0, 0, 0, 1, 0], dtype=int)  # A -> B
        self.reaction_B_to_C = np.array([0, 1, 0, 0, 0, 1], dtype=int)  # B -> C
        self.reaction_C_to_A = np.array([0, 0, 1, 1, 0, 0], dtype=int)  # C -> A
        self.reaction_AB_to_C = np.array([1, 1, 0, 0, 0, 1], dtype=int)  # A + B -> C

        # Create a reaction pool
        self.reaction_list = [
            self.reaction_A_to_B,
            self.reaction_B_to_C,
            self.reaction_C_to_A,
            self.reaction_AB_to_C
        ]

        # Create initial mechanisms
        self.mech1 = [self.reaction_A_to_B, self.reaction_B_to_C] # A->B->C
        self.mech2 = [self.reaction_C_to_A, self.reaction_AB_to_C]  # C->A + A+B->C

        # Parameters for testing
        self.UPPER_LIMIT = 4
        self.LOWER_LIMIT = 1
        self.NUM_NEW_MECHS = 2
        self.selection_pressure = 0.2
        self.mut_prob = 0.1


    def test_crossover_basic(self):
        """Test that crossover produces valid offspring networks"""
        offspring1, offspring2 = crossover(
            self.mech1,
            self.mech2,
            self.UPPER_LIMIT,
            self.LOWER_LIMIT
        )

        # Check basic properties
        self.assertIsInstance(offspring1, list)
        self.assertIsInstance(offspring2, list)
        self.assertTrue(1 <= len(offspring1) <= self.UPPER_LIMIT)
        self.assertTrue(1 <= len(offspring2) <= self.UPPER_LIMIT)

        # Check all reactions are from the parent pool
        all_reactions = self.mech1 + self.mech2
        for rxn in offspring1 + offspring2:
            self.assertTrue(any(np.array_equal(rxn, x) for x in all_reactions))

        # Check validation (all species should be present)
        num_species = len(self.mech1[0]) // 2
        self.assertTrue(validate_mechanism(offspring1, num_species))
        self.assertTrue(validate_mechanism(offspring2, num_species))


    def test_crossover_edge_cases(self):
        """Test crossover with edge case inputs"""
        # Test with identical parents
        offspring1, offspring2 = crossover(
            self.mech1,
            self.mech1,
            self.UPPER_LIMIT,
            self.LOWER_LIMIT
        )
        self.assertEqual(len(offspring1), len(offspring2))

        # Test with minimum size limit
        offspring1, offspring2 = crossover(
            self.mech1,
            self.mech2,
            self.UPPER_LIMIT,
            LOWER_LIMIT=2  # Force minimum size of 2
        )
        self.assertTrue(len(offspring1) >= 2)
        self.assertTrue(len(offspring2) >= 2)


    def test_validate_mechanism(self):
        """Test network validation function"""
        num_species = len(self.mech1[0]) // 2

        # Valid network (contains all species)
        valid_network = [self.reaction_A_to_B, self.reaction_B_to_C]  # A->B->C
        self.assertTrue(validate_mechanism(valid_network, num_species))

        # Invalid network (missing species C)
        invalid_network = [self.reaction_A_to_B]  # Only A->B
        self.assertFalse(validate_mechanism(invalid_network, num_species))

        # Edge case: empty network
        self.assertFalse(validate_mechanism([], num_species))


    def test_next_generation_mechanism(self):
        """Test the main next generation function"""
        # Create mock sorted mechanism IDs (higher index = better fitness)
        mech_ids_sorted = [(0,), (1,)]  # mech1 is better than mech2

        # Previous generation mechanisms
        prev_gen_mechs = [self.mech1, self.mech2]

        # Generate new mechanisms
        new_mechs = next_generation_mechanism(
            mech_ids_sorted,
            [self.mech1, self.mech2],
            prev_gen_mechs,
            self.NUM_NEW_MECHS,
            self.selection_pressure,
            self.UPPER_LIMIT,
            self.LOWER_LIMIT
        )

        # Check basic properties
        self.assertEqual(len(new_mechs), self.NUM_NEW_MECHS)
        for mech in new_mechs:
            self.assertTrue(self.LOWER_LIMIT <= len(mech) <= self.UPPER_LIMIT)
            self.assertFalse(mech_check(prev_gen_mechs, mech))  # Should be unique


    def test_generate_mutations(self):
        """Test mutation generation function"""
        # Create a copy of the mechanisms to mutate
        mech_list = [self.mech1.copy(), self.mech2.copy()]
        original_mech_list = [m.copy() for m in mech_list]

        # Apply mutations
        generate_mutations(mech_list, self.mut_prob, self.reaction_list)

        # Check if mutations occurred (probabilistic, so we can't be certain)
        changed = False
        for original, mutated in zip(original_mech_list, mech_list):
            if len(original) != len(mutated):
                changed = True
                break
            for r1, r2 in zip(original, mutated):
                if not np.array_equal(r1, r2):
                    changed = True
                    break

        # At least verify the function didn't break anything
        for mech in mech_list:
            num_species = len(mech[0]) // 2
            self.assertTrue(validate_mechanism(mech, num_species))


    def test_array_not_in_list(self):
        """Test array presence checking function"""
        arr_list = [np.array([1, 0]), np.array([0, 1])]

        # Test with array not in list
        test_arr = np.array([1, 1])
        self.assertTrue(array_not_in_list(test_arr, arr_list))

        # Test with array in list
        test_arr = np.array([1, 0])
        self.assertFalse(array_not_in_list(test_arr, arr_list))


    def test_species_in_reaction(self):
        """Test species participation tracking"""
        num_species = 3
        initial_state = np.zeros(num_species, dtype=bool)

        # Test with a reaction that involves species 0 and 1
        updated = species_in_reaction(initial_state, self.reaction_A_to_B, num_species)
        self.assertTrue(updated[0])  # Species A
        self.assertTrue(updated[1])  # Species B
        self.assertFalse(updated[2])  # Species C not in reaction


    def test_mech_check(self):
        """Test mechanism uniqueness checking"""
        mech_list = [self.mech1, self.mech2]

        # Test with identical mechanism
        self.assertTrue(mech_check(mech_list, self.mech1))

        # Test with different mechanism
        new_mech = [self.reaction_A_to_B, self.reaction_C_to_A]
        self.assertFalse(mech_check(mech_list, new_mech))

        # Test with permuted version (should be considered equal)
        permuted_mech = [self.mech1[1], self.mech1[0]]  # reversed order
        self.assertTrue(mech_check(mech_list, permuted_mech))


if __name__ == '__main__':
    unittest.main()
