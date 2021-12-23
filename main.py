import sys
sys.path.append("src/main")
import src.main.utils as utils
import src.main.utils_ml as ml_utils
import numpy as np

"""
    main script to test rule-based system or ml-algorithm
"""

BDL = "data/bdl_a" #man
SLT = "data/slt_a" #woman
RMS = "data/rms_a" #man
CMS = "data/cms_a" #woman

#Here we can test rule-based system on different speaker folder
utils.system_01(BDL)
utils.system_02(CMS)

#Here we can test ml model
model, scaler = ml_utils.load_BinaryClassificationModel("data/ml_data/BinaryClassificationModel.pt", "data/ml_data/BinaryClassificationScaler.pt")
features= utils.analyse(CMS)
data = [0, 0, 0, 0]
data[0] = features[0]
data[1] = features[1]
data[2] = np.mean(features[2])
data[3] = np.mean(features[3])
ml_utils.useModel(model, scaler, data)