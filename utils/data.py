import os
import time
import numpy as np
import scipy.io as sio
import random
import torch
import datetime
import nibabel as nib

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


def load_nii(filename):
    return nib.load(filename).get_data()


def save_nii(data, filename, filename_sample=''):
    if filename_sample:
        nib.save(nib.Nifti1Image(data, None, nib.load(filename_sample).header), filename)
    else:
        nib.save(nib.Nifti1Image(data, None, None), filename)


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
        print(filename)
        data = load_h5(filename, varname=varname)
        if data.ndim == 4:
            data = data.transpose(3,2,1,0)
        elif data.ndim == 3:
            data = data.transpose(2,1,0)
    return data


def save_mat(filename, varname, data):
    adict = {}
    adict[varname] = data
    sio.savemat(filename, adict)


def readcfl(name):
    # get dims from .hdr
    h = open(name + ".hdr", "r")
    h.readline() # skip
    l = h.readline()
    h.close()
    dims = [int(i) for i in l.split( )]

    # remove singleton dimensions from the end
    n = np.prod(dims)
    dims_prod = np.cumprod(dims)
    dims = dims[:np.searchsorted(dims_prod, n)+1]

    # load data and reshape into dims
    d = open(name + ".cfl", "r")
    a = np.fromfile(d, dtype=np.complex64, count=n)
    d.close()
    return a.reshape(dims, order='F') # column-major

	
def writecfl(name, array):
    h = open(name + ".hdr", "w")
    h.write('# Dimensions\n')
    for i in (array.shape):
            h.write("%d " % i)
    h.write('\n')
    h.close()
    d = open(name + ".cfl", "w")
    array.T.astype(np.complex64).tofile(d) # tranpose for column-major order
    d.close()


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


def r2c(img, flag_me=0):
    """
    for both images with and without batch dim, return an image with batch dim
    """
    if img.dtype == 'float32':
        dtype = np.complex64
    else:
        dtype = np.complex128

    if flag_me == 0:
        if len(img.shape) == 3:
            img = img[np.newaxis, ...]
        # out = np.zeros((img.shape[0], img.shape[2], img.shape[3]), dtype=dtype)
        out = np.zeros(img.shape[0:1] + img.shape[2:], dtype=dtype)
    
    elif flag_me == 1:
        # img = np.concatenate((img[:, np.newaxis, :10, ...], img[:, np.newaxis, 10:, ...]), axis=1)

        img_old = img
        img = np.zeros(img.shape[0:1] + (2, img.shape[1]//2) + img.shape[2:], dtype=dtype)
        img[:, 0, ...] = img_old[:, 0::2, ...]
        img[:, 1, ...] = img_old[:, 1::2, ...]
        out = np.zeros((img.shape[0], img.shape[2], img.shape[3], img.shape[4]), dtype=dtype)
    
    if img.shape[1] == 1:
        out = img[:, 0, ...]
    else:
        out = img[:, 0, ...] + 1j*img[:, 1, ...]

    return out


def c2r(img, flag_me=0):
    """
    for single image, no batch dim
    flag_me: flag to concatenate echo dimension into the channel dimension
    """
    dtype = np.float32
    if flag_me == 0:
        # out = np.zeros((2, img.shape[0], img.shape[1]), dtype=dtype)
        out = np.zeros((2,) + img.shape, dtype=dtype)
        out[0, ...] = img.real
        out[1, ...] = img.imag
    else:
        out = np.zeros((2*img.shape[2], img.shape[0], img.shape[1]), dtype=dtype)
        out[0::2, ...] = np.moveaxis(img.real, -1, 0)
        out[1::2, ...] = np.moveaxis(img.imag, -1, 0)
    return out


def c2r_kdata(kdata):
    """
    for multi-coil kdata, no batch dim
    """
    dtype = np.float32
    out = np.zeros((kdata.shape+(2,)), dtype=dtype)
    out[..., 0] = kdata.real
    out[..., 1] = kdata.imag
    return out


def torch_channel_concate(img, necho=10):
    """
        concatenate the echo dim (2nd) to the channel dim (1st)
        output: (batch, 2*necho, nrow, ncol)
    """
    # device = img.get_device()
    # out = torch.empty(img.shape[0:1] + (2*img.shape[2],) + img.shape[3:]).to(device)
    # out[:, 0::2, ...] = img[:, 0, ...]
    # out[:, 1::2, ...] = img[:, 1, ...]

    # # option1
    # # out = torch.cat([img[:, 0, ...], img[:, 1, ...]], 1)

    # option2
    out = img[:, :, 0, ...]
    for i in range(1, necho):
        out = torch.cat([out, img[:, :, i, ...]], dim=1)
    return out


def torch_channel_deconcate(img):
    """
        deconcatenate the echo dim (2nd) back from the channel dim (1st)
        output: (batch, 2, necho, nrow, ncol)
    """
    # device = img.get_device()
    # out = torch.empty(img.shape[0:1] + (2, img.shape[1]//2) + img.shape[2:]).to(device)
    # out[:, 0, ...] = img[:, 0::2, ...]
    # out[:, 1, ...] = img[:, 1::2, ...]

    # # option1
    # out = torch.cat([img[:, None, :10, ...], img[:, None, 10:, ...]], dim=1)

    # option2
    out1 = img[:, 0::2, ...]
    out2 = img[:, 1::2, ...]
    out = torch.cat([out1[:, None, ...], out2[:, None, ...]], dim=1)
    return out


def torch_channel_to_complex(img):
    """
        convert multi-echo images with concatenated echos as channels to the complex images
    """
    real = img[:, 0::2, ...]
    imag = img[:, 1::2, ...]
    out = torch.complex(real, imag)
    return out
    

def cplx_mlpy(a, b):
    """
    multiply two 'complex' tensors (with the last dim = 2, representing real and imaginary parts)
    """
    # device = a.get_device()
    # out = torch.empty(a.shape).to(device)
    # out[..., 0] = a[..., 0]*b[..., 0] - a[..., 1]*b[..., 1]
    # out[..., 1] = a[..., 0]*b[..., 1] + a[..., 1]*b[..., 0]

    out1 = a[..., 0:1]*b[..., 0:1] - a[..., 1:2]*b[..., 1:2]
    out2 = a[..., 0:1]*b[..., 1:2] + a[..., 1:2]*b[..., 0:1]
    out = torch.cat([out1, out2], dim=-1)
    return out

def cplx_matmlpy(A, B):
    '''
    multiply two "complex" matrices (with the last dim = 2, representing real and imaginary parts)
    A: (k, M, 2), B: (M, N, 2)
    return C: (k, N, 2)
    '''
    # A = torch.repeat_interleave(A[:, :, None, :], B.size()[1], dim=2)
    # B = torch.repeat_interleave(B[None, :, :, :], A.size()[0], dim=0)
    # out1 = torch.sum(A[..., 0]*B[..., 0] - A[..., 1]*B[..., 1], dim=1, keepdim=False)
    # out2 = torch.sum(A[..., 0]*B[..., 1] + A[..., 1]*B[..., 0], dim=1, keepdim=False)

    out1 = torch.matmul(A[..., 0], B[..., 0]) - torch.matmul(A[..., 1], B[..., 1])
    out2 = torch.matmul(A[..., 0], B[..., 1]) + torch.matmul(A[..., 1], B[..., 0])
    out = torch.cat([out1[..., None], out2[..., None]], dim=-1)

    # test
    # device = A.get_device()
    # out = torch.ones(A.size()[0], B.size()[1], 2).to(device)
    return out

def my_isnan(a, i):
    print('K = {0}, {1}'.format(i, torch.sum(a[torch.isnan(a)])))
    return torch.sum(a[torch.isnan(a)])


def cplx_dvd(a, b):
    """
    division between a and b
    """
    # device = a.get_device()

    # denom = torch.empty(a.shape).to(device)
    # denom[..., 0] = b[..., 0]**2 + b[..., 1]**2
    # denom[..., 1] = denom[..., 0]

    # out = torch.empty(a.shape).to(device)
    # out[..., 0] = a[..., 0]*b[..., 0] + a[..., 1]*b[..., 1]
    # out[..., 1] = -a[..., 0]*b[..., 1] + a[..., 1]*b[..., 0]
    # out = out/denom

    denom0 = cplx_mlpy(cplx_conj(b), b)[..., 0:1]
    denom = torch.cat([denom0, denom0], dim=-1)
    out = cplx_mlpy(cplx_conj(b), a) / denom
    return out


def cplx_conj(a):
    """
    conjugate of a complex number
    """
    # device = a.get_device()
    # out = torch.empty(a.shape).to(device)
    # out[..., 0] = a[..., 0]
    # out[..., 1] = -a[..., 1]

    out = torch.cat([a[..., 0:1], -a[..., 1:2]], dim=-1)
    return out

def cplx_matconj(A):
    '''
    conjugate of a complex matrix
    A: (M, N, 2)
    return: (N, M, 2) with conjugate imaginary part 
    '''
    return cplx_conj(A.permute(1, 0, 2))

def fft_shift_row(image, nrows, flag_me=0):
    if flag_me == 0:
        return torch.cat((image[:, :, nrows//2:nrows, ...], image[:, :, 0:nrows//2, ...]), dim=2)
    else:
        return torch.cat((image[:, :, :, nrows//2:nrows, ...], image[:, :, :, 0:nrows//2, ...]), dim=3)


def fft_shift_col(image, ncols, flag_me=0):
    if flag_me == 0:
        return torch.cat((image[:, :, :, ncols//2:ncols, ...], image[:, :, :, 0:ncols//2, ...]), dim=3)
    else:
        return torch.cat((image[:, :, :, :, ncols//2:ncols, ...], image[:, :, :, :, 0:ncols//2, ...]), dim=4)


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


def memory_pre_alloc(gpu_id):
    # pre-occupy the memory   
    total, used = os.popen(
        '"nvidia-smi" --query-gpu=memory.total,memory.used --format=csv,nounits,noheader'
            ).read().split('\n')[int(gpu_id)].split(',')
    
    total = int(total)
    used = int(used)

    print('Total memory is {0} MB'.format(total))
    print('Used memory is {0} MB'.format(used))

    max_mem = int(total*0.8)
    block_mem = max_mem - used
    
    x = torch.rand((256, 1024, block_mem)).cuda()
    x = torch.rand((2, 2)).cuda()


class Logger():
    def __init__(
        self, 
        rootName,
        opt, 
        flagFrint=True, 
        flagSave=True
    ):
        
        self.flagFrint = flagFrint
        self.flagSave = flagSave

        self.logName = 'logs'
        self.rootName = rootName

        bcrnn = opt['bcrnn']
        loss = opt['loss']
        K = opt['K']
        loupe = opt['loupe']
        ratio = opt['samplingRatio']
        solver = opt['solver']
        # dataset = opt['dataset_id']
        # mc_fusion = opt['mc_fusion']
        # t2redesign = opt['t2w_redesign']
        dataset = 0
        mc_fusion = 0
        t2redesign = 0

        if(not os.path.exists(self.rootName)):
            os.mkdir(self.rootName)
        self.logPath = os.path.join(self.rootName, self.logName)
        print(self.logPath)

        self.t0 = datetime.datetime.now().strftime("%I:%M%p on %B %d, %Y")
        print(self.t0)
        self.fileName = 'bcrnn={0}_loss={1}_K={2}_loupe={3}_ratio={4}_solver={5}_dataset={6}_mc_fusion={7}_t2redesign={8}'.format( \
                         bcrnn, loss, K, loupe, ratio, solver, dataset, mc_fusion, t2redesign) + '.log'
        self.filePath = os.path.join(self.logPath, self.fileName)

        if self.flagSave:
            if not os.path.exists(self.logPath):
                os.mkdir(self.logPath)
            self.file = open(self.filePath, 'w')
            self.file.write('Logs start:')
            self.file.write('\n')

    def print_and_save(self, string, *args):

        if not isinstance(string, str):
            string = str(string)

        if self.flagFrint:
            print(string % (args))

        if self.flagSave:
            self.file = open(self.filePath, 'a+')
            self.file.write(string % (args))
            self.file.write('\n')

    def close(self):

        # self.flagFrint = False
        # self.flagSave = False
        self.file.close()

    




