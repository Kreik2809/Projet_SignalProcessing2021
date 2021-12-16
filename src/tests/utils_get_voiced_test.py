import sys
sys.path.append("..")
sys.path.append("../main")
from main import utils
import unittest
import random
import numpy as np


class Get_voice_test(unittest.TestCase):
        
    """
        Unit test to test if there is at least one voice in frames
    """
    def test_01(self):
        current_signal, sampling_rate = utils.read_wavfile("../../data/bdl_a/arctic_a0001.wav")
        current_signal = utils.normalize(current_signal)
        frames = utils.split(current_signal, sampling_rate, 50, 25)
        voiced_segments, unvoiced_segments = utils.get_voiced(frames, 5)
        self.assertTrue(voiced_segments >= 1, "")
        
if __name__ == "__main__":
   unittest.main()