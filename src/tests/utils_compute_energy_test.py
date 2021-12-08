import sys
sys.path.append("..")
sys.path.append("../main")
from main import utils
import unittest
import random
import numpy as np


class Compute_energy_test(unittest.TestCase):
        
    """
        Unit test to test if the energy value is >= 0
    """
    def test_01(self):
        array = np.array([random.uniform(-10, 10) for p in range(10)])
        energy = utils.compute_energy(array)
        self.assertTrue(energy >= 0, "")
        
if __name__ == "__main__":
   unittest.main()