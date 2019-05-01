import numpy as np
import math
from utils.data import *


def psnr(img1, img2):
    """for normalized images (from 0 to 1) """
    mse = np.mean((img1 - img2)**2)
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
        outputs = normalization(abs(r2c(outputs)))
        targets = normalization(abs(r2c(targets)))
        for i in range(len(targets)):
            self.PSNRs.append(psnr(outputs[i], targets[i]))




