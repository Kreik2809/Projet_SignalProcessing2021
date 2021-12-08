import sys
sys.path.append("..")
sys.path.append("../main")
from main import utils
import unittest
import random
import numpy as np


class Normalize_test(unittest.TestCase):
    
    """
        Unit test to test if the max value of the normalized array is <= 1
    """
    def test_01(self):
        array = np.array([random.uniform(-10, 10) for p in range(10)])
        n_array = utils.normalize(array)
        maximum = max(n_array)
        self.assertTrue(maximum <= 1, "")
    
    """
        Unit test to test if the min value of the normalized array is >= -1
    """
    def test_02(self):
        array = np.array([random.uniform(-10, 10) for p in range(10)])
        n_array = utils.normalize(array)
        minimum = min(n_array)
        self.assertTrue(minimum >= -1, "")

if __name__ == "__main__":
   unittest.main()