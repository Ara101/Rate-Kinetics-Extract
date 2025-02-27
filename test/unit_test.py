import unittest
import numpy as np
from your_module import KineticAnalysis  # Replace with the actual module name

class TestKineticAnalysis(unittest.TestCase):

    def setUp(self):
        self.time = np.array([0, 1, 2, 3, 4, 5])
        self.response = np.array([0, 1, 2, 2.5, 2.8, 3])
        self.kinetic = KineticAnalysis(self.time, self.response)

    def test_baseline_steadystate_response(self):
        result = self.kinetic.baseline_steadystate_response(2, 0, 10, 1)
        expected = 10 * (1 - np.exp(-1 * 2))  # y_final * (1 - exp(-kon * t))
        self.assertAlmostEqual(result, expected, places=5)

    def test_response_to_zero(self):
        result = self.kinetic.response_to_zero(2, 5, 1, 2, 1)
        expected = (5 / (2 - 1)) * (np.exp(-1 * 2) - np.exp(-2 * 2)) + 1
        self.assertAlmostEqual(result, expected, places=5)

    def test_response_to_steady_state(self):
        result = self.kinetic.response_to_steady_state(2, 1, 10, 0.5, 2, 1)
        expected = 10 * (1 - 0.5 * np.exp(-2 * 2) + (0.5 - 1) * np.exp(-1 * 2)) + 1
        self.assertAlmostEqual(result, expected, places=5)

    def test_typical_association(self):
        result = self.kinetic.typical_association(2, 10, 5, 2, 1)
        kd = 1 / 2
        expected = ((10 * 5) / (kd + 5)) * (1 - np.exp((-1 * (2 * 5 + 1)) * 2))
        self.assertAlmostEqual(result, expected, places=5)

    def test_typical_dissociation(self):
        result = self.kinetic.typical_dissociation(2, 10, 1)
        expected = 10 * np.exp(-1 * 2)
        self.assertAlmostEqual(result, expected, places=5)

if __name__ == '__main__':
    unittest.main()