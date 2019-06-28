import os
import time
import numpy as np
import scipy.io as sio
import random
import torch
import datetime
from IPython.display import display
from PIL import Image


def recursiveFilesWithSuffix(rootDir = '.', suffix = ''):

    return [os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootDir)
        for filename in filenames if filename.endswith(suffix)]


def listFolders(rootDir = '.'):

    return [os.path.join(rootDir, filename)
        for filename in os.listdir(rootDir) if os.path.isdir(os.path.join(rootDir, filename))]


def listFilesWithSuffix(rootDir = '.', suffix = None):

    if suffix:
        res = [os.path.join(rootDir, filename) 
            for filename in os.listdir(rootDir) 
                if os.path.isfile(os.path.join(rootDir, filename)) and filename.endswith(suffix)]    
    else:
        res = [os.path.join(rootDir, filename) 
            for filename in os.listdir(rootDir) 
                if os.path.isfile(os.path.join(rootDir, filename))]    
    return res


def load_h5(filename, varname='data'):

    import h5py
    with h5py.File(filename, 'r') as f:
        data = f[varname][:]
    return data


def load_mat(filename, varname='data'):

    try:
        f = sio.loadmat(filename)
        data = f[varname]        
    except:
        data = load_h5(filename, varname=varname)
        if data.ndim == 4:
            data = data.transpose(3,2,1,0)
        elif data.ndim == 3:
            data = data.transpose(2,1,0)
    return data


def div0(a, b):
    """handling division by zero"""
    c = np.divide(a, b, out=np.zeros_like(a), where=b!=0)
    return c


def normalization(img):
    """ normalizing images (batchsize*nrow*ncol)"""
    img_new = np.empty(img.shape, dtype=img.dtype)
    for i in range(len(img)):
        img_new[i] = div0(img[i]-img[i].min(), img[i].ptp())
    return img_new



def r2c(img):
    """
    for both images with and without batch dim, return an image with batch dim
    """
    if img.dtype == 'float32':
        dtype = np.complex64
    else:
        dtype = np.complex128

    if len(img.shape) == 3:
        img = img[np.newaxis, ...]

    out = np.zeros((img.shape[0], img.shape[2], img.shape[3]), dtype=dtype)
    if img.shape[1] == 1:
        out = img[:, 0, ...]
    else:
        out = img[:, 0, ...] + 1j*img[:, 1, ...]
    return out


def c2r(img):
    """
    for single image, no batch dim
    """
    if img.dtype == 'complex64':
        dtype = np.float32
    else:
        dtype = np.float64
    
    out = np.zeros((2, img.shape[0], img.shape[1]), dtype=dtype)
    out[0, ...] = img.real
    out[1, ...] = img.imag
    return out


def cplx_mlpy(a, b):
    """
    multiply two 'complex' tensors (with the last dim = 2, representing real and imaginary parts)
    """
    device = a.get_device()
    out = torch.empty(a.shape).to(device)
    out[..., 0] = a[..., 0]*b[..., 0] - a[..., 1]*b[..., 1]
    out[..., 1] = a[..., 0]*b[..., 1] + a[..., 1]*b[..., 0]
    return out


def cplx_dvd(a, b):
    """
    division between a and b
    """
    device = a.get_device()

    denom = torch.empty(a.shape).to(device)
    denom[..., 0] = b[..., 0]**2 + b[..., 1]**2
    denom[..., 1] = denom[..., 0]

    out = torch.empty(a.shape).to(device)
    out[..., 0] = a[..., 0]*b[..., 0] + a[..., 1]*b[..., 1]
    out[..., 1] = -a[..., 0]*b[..., 1] + a[..., 1]*b[..., 0]
    out = out/denom
    return out


def cplx_conj(a):
    """
    conjugate of a complex number
    """
    device = a.get_device()
    out = torch.empty(a.shape).to(device)
    out[..., 0] = a[..., 0]
    out[..., 1] = -a[..., 1]
    return out


def showImage(img, idxs=[1,2,3,4,5], numShow=5, sampling=False):

    img = np.asarray(img)
    if len(img.shape) == 4:
        img = r2c(img)
        img = abs(img)

    numShow = np.amin((numShow, img.shape[0]))

    batch_size, nrow, ncol = img.shape[0:3]
    int_img = (img*255).clip(0, 255).astype('uint8')
    int_img = np.repeat(int_img[..., np.newaxis], 3, axis=3)
    imgstack = np.ones((nrow, numShow*ncol, 3)).astype('uint8')

    if sampling:
        idxs = random.sample(range(batch_size), numShow)
    else:
        idxs = idxs

    for x in range(len(idxs)):
        idx = idxs[x]
        imgstack[:, x*ncol:(x+1)*ncol, :] = int_img[idx, ...]

    display(Image.fromarray(imgstack))
    return imgstack, idxs


class Logger():

    def __init__(self, folderName, rootName, flagFrint=True, flagSave=True):
        
        self.flagFrint = flagFrint
        self.flagSave = flagSave

        self.folderName = folderName
        self.rootName = rootName

        if(not os.path.exists(self.rootName)):
            os.mkdir(self.rootName)
        self.logPath = os.path.join(self.rootName, self.folderName)

        self.t0 = datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")
        print(self.t0)
        self.fileName = 'logs_begining_at_' + self.t0 + '.log'
        self.filePath = os.path.join(self.logPath, self.fileName)

        if self.flagSave:
            if not os.path.exists(self.logPath):
                os.mkdir(self.logPath)
            self.file = open(self.filePath, 'w')

    def print_and_save(self, string, *args):

        if not isinstance(string, str):
            string = str(string)

        if self.flagFrint:
            print(string % (args))

        if self.flagSave:
            self.file = open(self.filePath, 'w')
            self.file.write(string % (args))
            self.file.write('\n')

    def close(self):

        # self.flagFrint = False
        # self.flagSave = False
        self.file.close()


    




