"""
utilities.py - utility module with utility functions and constants
"""
import os
import errno
from multiprocessing import cpu_count
from cfelpyutils.crystfel_utils import load_crystfel_geometry
from cfelpyutils.geometry_utils import apply_geometry_to_data

HIGH_GAIN = 0
MEDIUM_GAIN = 1
LOW_GAIN = 2
DATA_KEY = 'data'
GAIN_KEY = 'gain'
PULSE_KEY = 'pulseId'
TRAIN_KEY = 'trainId'
BASE_PATH = "/gpfs/exfel/u/scratch/MID/201802/p002200/cheetah/hdf5/r{0:04d}-data/XFEL-r{0:04d}-c{1:02d}.h5"
USER_PATH = os.path.join(os.path.dirname(__file__), "../../cheetah/XFEL-r{0:04d}-c{1:02d}.cxi")
OUT_PATH = "hdf5/r{0:04d}/XFEL-r{0:04d}-c{1:02d}.h5"
CORES_COUNT = cpu_count()
BG_ROI = (slice(5000), slice(None))
PUPIL_ROI = (slice(750, 1040), slice(780, 1090))

AGIPD_GEOM = load_crystfel_geometry(os.path.join(os.path.dirname(__file__), "agipd.geom"))

def apply_agipd_geom(frame):
    return apply_geometry_to_data(frame, AGIPD_GEOM)

def make_output_dir(path):
    try:
        os.makedirs(path)
    except OSError as error:
        if error.errno != errno.EEXIST:
            raise OSError(error.errno, error.strerror, error.filename)
