import sys
sys.path.append("..")
sys.path.append("../main")
from main import utils
import unittest
import random
import numpy as np


class Read_wavfile_test(unittest.TestCase):
    
    """
        Unit test to test if the ndarray size of the signal is > 0
    """
    def test_01(self):
        signal, sampling_rate = utils.read_wavfile("../../data/bdl_a/arctic_a0001.wav")
        self.assertTrue(signal.size > 0, "")
    
    """
        Unit test to test if sampling rate of the signal is > 0
    """
    def test_02(self):
        signal, sampling_rate = utils.read_wavfile("../../data/bdl_a/arctic_a0001.wav")
        self.assertTrue(sampling_rate > 0, "")

if __name__ == "__main__":
   unittest.main()