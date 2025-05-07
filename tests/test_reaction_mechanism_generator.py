import numpy as np
import unittest
from unittest.mock import patch
from unittest.mock import Mock
from src.SISR.reaction_mechanism_generator import ReactionMechanismGenerator
from src.SISR.reaction_mechanism_generator import (
    reactant_concentration,
    reaction_system,
    estimate_k_linearly,
    array_not_in_list
)

class TestReactionMechanismGenerator(unittest.TestCase):

    def setUp(self):
        """
        Set up for test methods.
        """
        self.generator = ReactionMechanismGenerator(order=1, num_generations=1, num_mech_per_gen=10)
        # Define sample reactions for testing
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
        self.mech1 = [self.reaction_A_to_B, self.reaction_B_to_C]  # A->B->C
        self.mech2 = [self.reaction_C_to_A, self.reaction_AB_to_C]  # C->A + A+B->C

        # Parameters for testing
        self.S = np.array([[1.0, 2.0, 0.5], [0.8, 1.8, 0.6],[0.6, 1.5, 0.7]])
        self.X = np.array([0, 1, 2])
        self.S_dot = calculated_derivatives = np.gradient(self.S, self.X, axis=0)
        self.S_dot_max = np.max(np.abs(calculated_derivatives), axis=0, keepdims=True)

        # RMG parameters
        self.generator.MAX_RXN = 2
        self.generator.MIN_RXN = 1
        self.generator.N_TARGETS = 3


    def test_initialization(self):
        """
        Test that the ReactionMechanismGenerator initializes correctly.
        """
        self.assertEqual(self.generator.ORDER, 1)
        self.assertEqual(self.generator.NUM_GENS, 1)
        self.assertEqual(self.generator.NUM_MECHS, 10)
        self.assertEqual(self.generator.PREV_GEN, 0.2)
        self.assertTrue(self.generator.include_bias)


    def test_invalid_initialization(self):
        """
        Test that the ReactionMechanismGenerator raises ValueError on invalid input.
        """
        with self.assertRaises(ValueError):
            ReactionMechanismGenerator(order=-1)
        with self.assertRaises(ValueError):
            ReactionMechanismGenerator(max_ratio=0)
        with self.assertRaises(ValueError):
            ReactionMechanismGenerator(min_rxns_per_mech = 2,max_rxns_per_mech = 1)


    def test_generate_reactions(self):
        """
        Test the _generate_reactions method.
        """
        # Set parameters for testing
        self.generator.MAX_STOICH = 1
        reactions = self.generator._generate_reactions()

        # Test 1: Check that the reactions are generated correctly
        self.assertIsNotNone(reactions)
        self.assertTrue(len(reactions) > 0)

        # Test 2: Check that the reactions are in the expected format
        for reaction in reactions:
            self.assertIsInstance(reaction, np.ndarray)
            self.assertEqual(reaction.shape[0], 2 * self.generator.N_TARGETS)
            self.assertTrue(np.all((reaction[:self.generator.N_TARGETS] >= 0) & (reaction[:self.generator.N_TARGETS] <= self.generator.ORDER)))
            self.assertTrue(np.all((reaction[self.generator.N_TARGETS:] >= 0) & (reaction[self.generator.N_TARGETS:] <= self.generator.MAX_STOICH*self.generator.ORDER)))

        # Test 3: Check that all reactions are unique
        unique_reactions = set(tuple(r) for r in reactions)
        self.assertEqual(len(unique_reactions), len(reactions))

        # Test 4: Check edge cases: N_TARGETS = 0
        self.generator.N_TARGETS = 0
        reactions = self.generator._generate_reactions()
        self.assertEqual(len(reactions), 0)

        # Test 5: Check edge cases: N_TARGETS = 1
        self.generator.N_TARGETS = 1
        reactions = self.generator._generate_reactions()
        self.assertTrue(len(reactions) > 0)

    def test_initialize_population(self):
        """
        Test the _initialize_population method.
        """
        # Call the function
        population = self.generator._initialize_population(self.reaction_list)

        # Test 1: Check that the population is generated correctly
        self.assertIsNotNone(population)
        self.assertTrue(len(population) <= self.generator.NUM_MECHS)

        # Test 2: Check that the population is in the expected format
        for mechanism in population:
            self.assertIsInstance(mechanism, list)
            self.assertTrue(len(mechanism) > 0)
            for reaction in mechanism:
                    self.assertIsInstance(reaction, np.ndarray)
                    self.assertEqual(reaction.shape[0], 2 * self.generator.N_TARGETS)
                    self.assertTrue(np.all((reaction[:self.generator.N_TARGETS] >= 0) & (reaction[:self.generator.N_TARGETS] <= self.generator.ORDER)))
                    self.assertTrue(np.all((reaction[self.generator.N_TARGETS:] >= 0) & (reaction[self.generator.N_TARGETS:] <= self.generator.MAX_STOICH*self.generator.ORDER)))


    def test_construct_matrix(self):
        """
        Test the _construct_matrix method.
        """
        # Call the function
        matrix = self.generator._construct_matrix(self.mech1)

        # Check the results
        self.assertIsNotNone(matrix)
        self.assertEqual(np.array(matrix).shape[0], 2)
        self.assertEqual(np.array(matrix).shape[1], 2 * self.generator.N_TARGETS)


    def test_fit_coefficients(self):
        """
        Test the _fit_coefficients method.
        """
        # Mock k value
        k = np.array([1, 2, 3])

        # Test 1: Basic functionality
        k_fit, mse = self.generator._fit_coefficients(
            S=self.S,
            reaction_mechanism=self.mech1,
            S_dot=self.S_dot,
            S_dot_max=self.S_dot_max,
            time=self.X
        )

        # Should return coefficients for each reaction
        self.assertEqual(len(k_fit), len(self.mech1))

        # MSE should be non-negative
        self.assertGreaterEqual(mse, 0)

        # Coefficients should be positive
        self.assertTrue(all(k > 0 for k in k_fit))

        # Test 2: Test with initial guess
        k0 = [0.5, 0.5]
        k_fit, mse = self.generator._fit_coefficients(
            S=self.S,
            reaction_mechanism=self.mech1,
            S_dot=self.S_dot,
            S_dot_max=self.S_dot_max,
            time=self.X,
            k0=k0
        )

        # Should return coefficients for each reaction
        self.assertEqual(len(k_fit), len(k0))

        # Test 3: Test with empty S, should raise ValueError
        with self.assertRaises(ValueError):
            self.generator._fit_coefficients(
                S=np.array([]),
                reaction_mechanism=self.mech1,
                S_dot=self.S_dot,
                S_dot_max=self.S_dot_max,
                time=self.X
            )


    def test_update_best_mechs(self):
        """
        Test the _update_best_mechs method.
        """
        # Mock data
        # mech1 and mech2 have different data
        mech1 = ([self.reaction_A_to_B, self.reaction_B_to_C], "metadata1", 0.1)
        mech2 = ([self.reaction_C_to_A, self.reaction_AB_to_C], "metadata2", 0.4)
        # mech3 has the same data as mech1 but in a different order
        mech3 = ([self.reaction_A_to_B, self.reaction_B_to_C], "metadata3", 0.5)
        # mech4 has different data than mech1 and mech2 but lower performance score than mech2
        mech4 = ([self.reaction_AB_to_C, self.reaction_C_to_A], "metadata4", 0.2)

        # Test 1: Basic functionality with unique mechanisms
        best_mechs = [mech1]
        new_mechs = [mech2]
        self.generator._update_best_mechs(best_mechs, new_mechs, 2)

        # Should contain both mech1 and mech2
        self.assertEqual(len(best_mechs), 2)
        self.assertEqual(best_mechs[0], mech1)  # Lower performance score is better
        self.assertEqual(best_mechs[1], mech2)

        # Test 2: Adding duplicate mechanism
        best_mechs = [mech1]
        new_mechs = [mech3]
        self.generator._update_best_mechs(best_mechs, new_mechs, 2)

        # Should only contain mech1
        self.assertEqual(len(best_mechs), 1)
        self.assertEqual(best_mechs[0], mech1)

        # Test 3: Empty inputs
        best_mechs = []
        new_mechs = []
        self.generator._update_best_mechs(best_mechs, new_mechs, 2)

        # Should remain empty
        self.assertEqual(len(best_mechs), 0)

        # Test 4: Best mechanisms already full
        best_mechs = [mech1, mech2]
        new_mechs = [mech4]
        self.generator._update_best_mechs(best_mechs, new_mechs, 2)

        # Should replace mech2 with mech4 since mech4 has a lower performance score
        self.assertEqual(len(best_mechs), 2)
        self.assertEqual(np.array(best_mechs[0][0]).all(), np.array(mech1[0]).all())
        self.assertEqual(np.array(best_mechs[1][0]).all(), np.array(mech4[0]).all())


    def test_evolve_population(self):
        """
        Test the _evolve_population method.
        """
        # Mock data
        self.generator.NUM_MECHS = 2
        self.generator.PREV_GEN = 0.5
        self.generator.N_TARGETS = 3
        population = [self.mech1,self.mech2]

        # Call the function
        best_mechs, ks, rm, mse = self.generator._evolve_population(
            population,
            self.S,
            self.X,
            self.reaction_list,
            "tests/test_rxn.txt",
            "tests/test_mech.txt")

        # Test 1: Check that the best_mechs are generated correctly
        self.assertIsNotNone(best_mechs)
        self.assertTrue(len(best_mechs) == int(self.generator.PREV_GEN * self.generator.NUM_MECHS))

        # Test 2: Check that ks, rm, mse are generated correctly
        self.assertIsNotNone(ks)
        self.assertIsNotNone(rm)
        self.assertTrue(len(ks) == len(rm))
        self.assertIsInstance(mse, float)

        # Test 3: Check best_mechs are in the expected format
        for mechanism in best_mechs:
            self.assertIsInstance(mechanism, list)
            self.assertTrue(len(mechanism) > 0)
            for reaction in mechanism[0]:
                self.assertIsInstance(reaction, np.ndarray)
                self.assertEqual(reaction.shape[0], 2 * self.generator.N_TARGETS)
                self.assertTrue(np.all((reaction[:self.generator.N_TARGETS] >= 0) & (reaction[:self.generator.N_TARGETS] <= self.generator.ORDER)))
                self.assertTrue(np.all((reaction[self.generator.N_TARGETS:] >= 0) & (reaction[self.generator.N_TARGETS:] <= self.generator.MAX_STOICH*self.generator.ORDER)))


    def test_reactant_concentration(self):
        """
        Test the reactant_concentration method.
        """
        reactants = np.array([1, 0])
        X = np.array([2, 3])
        value = reactant_concentration(reactants, X)

        # Test 1: Check that the concentration is calculated correctly
        self.assertEqual(value, 2)

        # Test 2: Check with zero reactants
        reactants = np.array([0, 0])
        value = reactant_concentration(reactants, X)
        self.assertEqual(value, 1.0)


    def test_reaction_system(self):
        """
        Test the reaction_system method.
        """
        # Mock data
        t = [0.0,0.1,0.2]
        rm = [self.reaction_A_to_B, self.reaction_B_to_C]
        ks = np.array([1.0, 2.0])

        # Call the function
        result = reaction_system(t, self.S, rm, ks)

        # Test 1: Check that the result is a numpy array
        self.assertIsInstance(result, np.ndarray)

        # Test 2: Check that the result has the same shape as y
        self.assertEqual(result.shape, self.S.shape)


    def test_estimate_k_linearly(self):
        """
        Test the estimate_k_linearly function.
        """
        # Mock data
        x = np.array([0, 1, 2])
        y = np.array([0, 1, 4])
        expected_slope = 2.0

        # Call the function
        slope = estimate_k_linearly(x, y)

        # Test 1: Check the results
        self.assertAlmostEqual(slope, expected_slope)

if __name__ == '__main__':
    unittest.main()
