import numpy as np
import math
from utils.data import *


def psnr(img1, img2):
    """for normalized images (from 0 to 1), img2 is the refernce ground truth"""
    mask = abs(img2) > 0.02
    mse = np.mean((img1 - img2)**2*mask)
    if mse == 0:
        return 100
    max_intensity = 1
    return 20*math.log10(max_intensity/math.sqrt(mse))
    

class Metrices():

    def __init__(self):
        self.PSNRs = []
        self.SSIMs = []
        self.NMSEs = []

    def get_metrices(self, outputs, targets):
        outputs = np.squeeze(np.asarray(outputs.cpu().detach()))
        targets = np.squeeze(np.asarray(targets.cpu().detach()))
        # outputs = normalization(abs(r2c(outputs)))
        # targets = normalization(abs(r2c(targets)))
        outputs = abs(r2c(outputs))
        targets = abs(r2c(targets))

        # outputs = normalization(outputs)
        # targets = normalization(targets)
        # weights = targets > 1e-1
        # outputs = outputs * weights
        # targets = targets * weights
        for i in range(len(targets)):
            self.PSNRs.append(psnr(outputs[i], targets[i]))

    


    





