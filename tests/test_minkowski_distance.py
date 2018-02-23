from unittest import TestCase
from bcs import minkowski_distance
import numpy as np

class TestMinkowski_distance(TestCase):
    def test_p_1(self):
        p = 1
        target = np.array([0.25, 0.25, 0.5])
        self.assertAlmostEqual(0.04, minkowski_distance(np.array([0.24, 0.24, 0.52]), target, p))
        self.assertAlmostEqual(0.2, minkowski_distance(np.array([0.22, 0.35, 0.43]), target, p))
