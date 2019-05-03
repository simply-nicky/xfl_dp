import os, errno
from cfelpyutils.crystfel_utils import load_crystfel_geometry
from cfelpyutils.geometry_utils import apply_geometry_to_data

basepath = "/gpfs/exfel/u/scratch/MID/201802/p002200/cheetah/hdf5/r{0:04d}-{2:s}/XFEL-r{0:04d}-c{1:02d}.{3:s}"
userpath = "XFEL-r{0:04d}-c{1:02d}.{2:s}"
outpath = "../hdf5/r{0:04d}-processed/XFEL-r{0:04d}-c{1:02d}-processed.{2:s}"
datapath = "entry_1/instrument_1/detector_1/detector_corrected/data"
trainpath = "/instrument/trainID"
pulsepath = "/instrument/pulseID"

AGIPD_geom = load_crystfel_geometry("/home/vmariani/Workspaces-refactored/crystallography/karabo/agipd.geom")

class worker_star(object):
    def __init__(self, worker):
        self.worker = worker
    
    def __call__(self, args):
        return self.worker(*args)

def get_path_to_data(rnum, cnum, tag, ext, online):
    return basepath.format(rnum, cnum, tag, ext) if online else userpath.format(rnum, cnum, ext)

def apply_agipd_geom(frame):
    return apply_geometry_to_data(frame, AGIPD_geom)

def make_output_dir_and_path(path):
    abspath = os.path.join(os.path.dirname(__file__), path)
    try:
        os.makedirs(os.path.dirname(abspath))
    except OSError as e:
        if e.errno != errno.EEXIST: raise
    return abspath