import h5py
from exfel import hg_gui_calibrate

OUT_PATH = "hdf5/r{0:04d}-processed/XFEL-r{0:04d}-c{1:02d}.cxi"
DATA_PATH = 'data/data/4'

def main(filename=OUT_PATH.format(283, 0), data_path=DATA_PATH):
    data = h5py.File(filename, 'r')[data_path][:]
    zero_adu, one_adu = hg_gui_calibrate(data.ravel(), filename)
    print("Zero ADU: {0:.1f}, One ADU: {1:.1f}".format(zero_adu, one_adu))

if __name__ == "__main__":
    main()
    