import unittest
from unittest.mock import patch
import numpy as np
from src.Reaction_Mechanism_Generator import ReactionMechanismGenerator

class TestReactionMechanismGenerator(unittest.TestCase):

    def setUp(self):
        """
        Set up for test methods.
        """
        self.generator = ReactionMechanismGenerator(order=1, num_generations=1, num_mech_per_gen=10)

    def test_initialization(self):
        """
        Test that the ReactionMechanismGenerator initializes correctly.
        """
        self.assertEqual(self.generator.order, 1)
        self.assertEqual(self.generator.num_gens, 1)
        self.assertEqual(self.generator.num_mechs, 10)
        self.assertTrue(self.generator.include_bias)

    def test_invalid_initialization(self):
        """
        Test that the ReactionMechanismGenerator raises ValueError on invalid input.
        """
        with self.assertRaises(ValueError):
            ReactionMechanismGenerator(order=-1)
        with self.assertRaises(ValueError):
            ReactionMechanismGenerator(max_ratio=0)

    def test_generate_reactions(self):
        """
        Test the _generate_reactions method.
        """
        self.generator.n_targets = 2
        reactions = self.generator._generate_reactions()
        self.assertIsNotNone(reactions)
        self.assertTrue(len(reactions) > 0)

    def test_species_in_rxn(self):
        """
        Test the species_in_rxn method.
        """
        species_included = np.array([False, False])
        reaction = np.array([1, 0, 0, 1])
        n_targets = 2
        updated_species = self.generator.species_in_rxn(species_included, reaction, n_targets)
        self.assertTrue(np.array_equal(updated_species, [True, True]))

    def test_reactant_concentration(self):
        """
        Test the reactant_concentration method.
        """
        reactants = np.array([1, 0])
        X = np.array([2, 3])
        concentration = self.generator.reactant_concentration(reactants, X)
        self.assertEqual(concentration, 2)

    @patch('Reaction_Mechanism_Generator.estimate_k_linearly')
    def test_generate_mechanism(self, mock_estimate_k_linearly):
        """
        Test the _generate_mechanism method.
        """
        mock_estimate_k_linearly.return_value = 1.0
        self.generator.n_targets = 2
        reactions_with_probs = self.generator._generate_reactions()
        Theta = np.array([[0.1, 0.2], [0.3, 0.4]])
        X = np.array([0, 1])
        mechanism_list = self.generator._generate_mechanism(reactions_with_probs, Theta, X)
        self.assertIsNotNone(mechanism_list)
        self.assertTrue(len(mechanism_list) > 0)

if __name__ == '__main__':
    unittest.main()
