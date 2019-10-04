import numpy as np
import h5py
from scipy.optimize import curve_fit

DATA_FOLDER = "/gpfs/exfel/exp/MID/201802/p002200/proc/r{0:04d}/CORR-R{0:04d}-AGIPD{1:02d}-S00001.h5"
DATA_PATH = "/INSTRUMENT/MID_DET_AGIPD1M-1/DET/{0:d}CH0:xtdf/image/"
OFFSETS = h5py.File('cheetah/calib/Cheetah-calib.h5', 'r')['AnalogOffset']
GAIN_LEVELS = h5py.File('cheetah/calib/Cheetah-calib.h5', 'r')['DigitalGainLevel']

def gauss(arg, amplitude, mu, sigma):
    """
    Gaussian function

    arg - argument
    amplitude - amplitude constant, max function value
    mu - center of gaussian
    sigma - gaussian width
    """
    return amplitude * np.exp(-(arg - mu)**2 / 2 / sigma**2)