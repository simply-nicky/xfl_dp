import h5py
from exfel import hg_gui_calibrate, RawData

CHEETAH_PATH = "hdf5/r{0:04d}-processed/XFEL-r{0:04d}-c{1:02d}.cxi"
CHEETAH_DATA_PATH = "data/data/4"
RAW_PATH = "raw/r0221_raw.h5"
RAW_DATA_PATH = "data/ADU"
RAW_GAIN_PATH = "data/DigitalGains"
RAW_PULSE_PATH = "data/pulseID"
RAW_TRAIN_PATH = "data/trainID"

def calib_cheetah(filename=CHEETAH_PATH.format(283, 0), data_path=CHEETAH_DATA_PATH):
    data = h5py.File(filename, 'r')[data_path][:]
    zero_adu, one_adu = hg_gui_calibrate(data.ravel(), filename)
    print("Zero ADU: {0:.1f}, One ADU: {1:.1f}".format(zero_adu, one_adu))

def calib_raw(filename=RAW_PATH,
              data_path=RAW_DATA_PATH,
              gain_path=RAW_GAIN_PATH,
              pulse_path=RAW_PULSE_PATH,
              train_path=RAW_TRAIN_PATH):
    raw_data = RawData(file_path=filename,
                       data_path=data_path,
                       gain_path=gain_path,
                       pulse_path=pulse_path,
                       train_path=train_path)
    data = raw_data.get_ordered_data(pids=4)
    print("Pulse ID: {0:d}, Data shape: {1}".format(data['pulseId'][0], data['data'].shape))
    hg_data = data['data'][data['gain'] == 0]
    print("HG_data shape: {}".format(hg_data.shape))
    one_adu, zero_adu = hg_gui_calibrate(hg_data, filename, full_roi=(-400, 6000))
    print("Zero ADU: {0:.1f}, One ADU: {1:.1f}".format(zero_adu, one_adu))

def save_raw(filename=RAW_PATH,
             data_path=RAW_DATA_PATH,
             gain_path=RAW_GAIN_PATH,
             pulse_path=RAW_PULSE_PATH,
             train_path=RAW_TRAIN_PATH):
    raw_data = RawData(file_path=filename,
                       data_path=data_path,
                       gain_path=gain_path,
                       pulse_path=pulse_path,
                       train_path=train_path)
    raw_data.save_ordered(pids=4)
    print("Data is saved at the location: {}".format(raw_data.out_path))

if __name__ == "__main__":
    save_raw()
