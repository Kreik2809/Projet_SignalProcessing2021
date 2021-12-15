import sys
sys.path.append("..")
sys.path.append("../main")
from main import utils
import unittest
import random
import numpy as np
import math


class Compute_formants_test(unittest.TestCase):
        
    """
        Unit test to test if 
    """
    def test_01(self):
        formants_bdl = utils.compute_formants("../../data/bdl_a/arctic_a0001.wav")
        formants_slt = utils.compute_formants("../../data/slt_a/arctic_a0001.wav")
        f1_list = []
        f2_list = []
        print("BDL (Homme) : ")
        for f in formants_bdl:
            print(f)
            f1 = f[1]
            f1_list.append(f1)
        print("SLT (Femme) : ")
        for f in formants_slt:
            print(f)
            f2 = f[1]
            f2_list.append(f2)
            
        print(np.mean(f1_list))
        print(np.mean(f2_list))

        
if __name__ == "__main__":
   unittest.main()