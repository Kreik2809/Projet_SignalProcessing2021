import sys
sys.path.append("..")
sys.path.append("../main")
from main import utils
import unittest
import random
import numpy as np


class Split_test(unittest.TestCase):
        
    """
        Unit test to test split function with sliding step equals to frame size
    """
    def test_01(self):
        tab = np.array([random.uniform(-10, 10) for p in range(100)])
        windows = utils.split(tab, 1000, 20, 20)
        self.assertTrue((tab[:20] == windows[0]).all(), "First frame test")
        self.assertTrue((tab[20:40] == windows[1]).all(), "Second frame test")
        self.assertFalse((tab[10:30] == windows[0]).all(), "Wrong frame test")

    """
        Unit test to test split function with sliding step < thant frame size
    """
    def test_02(self):
        tab = np.array([random.uniform(-10, 10) for p in range(100)])
        windows = utils.split(tab, 1000, 20, 10)
        self.assertTrue((tab[:20] == windows[0]).all(), "First frame test")
        self.assertTrue((tab[10:30] == windows[1]).all(), "Second frame test")
        self.assertFalse((tab[20:40] == windows[1]).all(), "Wrong frame test")

    def test_03(self):
        tab = np.array([random.uniform(-10, 10) for p in range(100)])
        windows = utils.split(tab, 1000, 20, 30)
        self.assertTrue((tab[:20] == windows[0]).all(), "First frame test")
        self.assertTrue((tab[30:50] == windows[1]).all(), "Second frame test")
        self.assertFalse((tab[20:40] == windows[1]).all(), "Wrong frame test")
        
if __name__ == "__main__":
   unittest.main()