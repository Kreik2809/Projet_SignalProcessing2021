import sys
sys.path.append("..")
sys.path.append("../main")
from main import utils
import unittest
import random
import numpy as np


class Cepstrum_pitch_estim_test(unittest.TestCase):
        
    """
        Unit test to test if the speaker is a man in the audio
    """
    def test_01(self):
        pitch_men = utils.cepstrum_pitch_estim("data/bdl_a")
        self.assertTrue(pitch_men >= 60, "")
        self.assertTrue(pitch_men <= 170, "")

    """
        Unit test to test if the speaker is a women in the audio
    """
    def test_02(self):
        pitch_women = utils.cepstrum_pitch_estim("data/slt_a")
        self.assertTrue(pitch_women >= 171, "")
        self.assertTrue(pitch_women <= 300, "")
        
if __name__ == "__main__":
   unittest.main()