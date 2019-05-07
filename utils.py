import os, errno, numpy as np
from cfelpyutils.crystfel_utils import load_crystfel_geometry
from cfelpyutils.geometry_utils import apply_geometry_to_data

basepath = "/gpfs/exfel/u/scratch/MID/201802/p002200/cheetah/hdf5/r{0:04d}-{2:s}/XFEL-r{0:04d}-c{1:02d}.{3:s}"
userpath = "XFEL-r{0:04d}-c{1:02d}.{2:s}"
outpath = "hdf5/r{0:04d}-processed/XFEL-r{0:04d}-c{1:02d}-processed.{2:s}"
datapath = "entry_1/instrument_1/detector_1/detector_corrected/data"
trainpath = "/instrument/trainID"
pulsepath = "/instrument/pulseID"
bg_roi = (slice(5000), slice(None))
thread_size = 100

AGIPD_geom = load_crystfel_geometry(os.path.join(os.path.dirname(__file__), "agipd.geom"))

class worker_star(object):
    def __init__(self, worker):
        self.worker = worker
    
    def __call__(self, args):
        return self.worker(*args)

def chunkify(start, end):
    thread_num = (end - start) // thread_size + 1
    return np.linspace(start, end, thread_num + 1).astype(int)

def get_path_to_data(rnum, cnum, tag, ext, online):
    return basepath.format(rnum, cnum, tag, ext) if online else userpath.format(rnum, cnum, ext)

def apply_agipd_geom(frame):
    return apply_geometry_to_data(frame, AGIPD_geom)

def make_output_dir(path):
    try:
        os.makedirs(os.path.dirname(path))
    except OSError as e:
        if e.errno != errno.EEXIST: raise

def output_path(rnum, cnum, ext, output_folder=os.path.dirname(os.path.dirname(__file__))):
    abspath = os.path.join(output_folder, outpath.format(rnum, cnum, ext))
    if not os.path.isfile(abspath):
        return abspath
    else:
        return output_path(rnum, cnum + 1, ext, output_folder)