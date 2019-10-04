"""
utilities.py - utility module with utility functions and constants
"""
import os
import errno
from multiprocessing import cpu_count
import numpy as np
from cfelpyutils.crystfel_utils import load_crystfel_geometry
from cfelpyutils.geometry_utils import apply_geometry_to_data

BASE_PATH = "/gpfs/exfel/u/scratch/MID/201802/p002200/cheetah/hdf5/r{0:04d}-data/XFEL-r{0:04d}-c{1:02d}.h5"
USER_PATH = os.path.join(os.path.dirname(__file__), "../../cheetah/XFEL-r{0:04d}-c{1:02d}.cxi")
OUT_PATH = "hdf5/r{0:04d}/XFEL-r{0:04d}-c{1:02d}.h5"
DATA_PATH = "entry_1/instrument_1/detector_1/detector_corrected/data"
TRAIN_PATH = "/instrument/trainID"
PULSE_PATH = "/instrument/pulseID"
CORES_COUNT = cpu_count()
BG_ROI = (slice(5000), slice(None))
PUPIL_ROI = (slice(750, 1040), slice(780, 1090))

GAINS = {68.8, 1.376}
GAIN_VERGE = 6000
AGIPD_GEOM = load_crystfel_geometry(os.path.join(os.path.dirname(__file__), "agipd.geom"))

def add_data_to_dset(dset, data):
    dset.refresh()
    dsetshape = dset.shape
    dset.resize((dsetshape[0] + data.shape[0],) + dsetshape[1:])
    dset[dsetshape[0]:] = data
    dset.flush()

def chunkify(start, end, thread_num=CORES_COUNT):
    limits = np.linspace(start, end, thread_num + 1).astype(int)
    return list(zip(limits[:-1], limits[1:]))

def apply_agipd_geom(frame):
    return apply_geometry_to_data(frame, AGIPD_GEOM)

def make_output_dir(path):
    try:
        os.makedirs(path)
    except OSError as error:
        if error.errno != errno.EEXIST:
            raise OSError(error.errno, error.strerror, error.filename)
