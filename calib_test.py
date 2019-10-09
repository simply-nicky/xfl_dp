from exfel import RawDataSplit, RawData, GainLevel, hg_gui_calibrate
from exfel.utils import HIGH_GAIN

RAW_FILE_PATH = "/gpfs/exfel/exp/MID/201802/p002200/raw/r0221/RAW-R0221-AGIPD{:02d}-S00000.h5"
PROC_FILE_PATH = "/gpfs/exfel/exp/MID/201802/p002200/proc/r0221/CORR-R0221-AGIPD{:02d}-S00000.h5"
DATA_PATH = "/INSTRUMENT/MID_DET_AGIPD1M-1/DET/{:d}CH0:xtdf/image/data"
GAIN_PATH = "/INSTRUMENT/MID_DET_AGIPD1M-1/DET/{:d}CH0:xtdf/image/gain"
TRAIN_PATH = "/INSTRUMENT/MID_DET_AGIPD1M-1/DET/{:d}CH0:xtdf/image/trainId"
PULSE_PATH = "/INSTRUMENT/MID_DET_AGIPD1M-1/DET/{:d}CH0:xtdf/image/pulseId"

def calib_raw(module_id=14,
              file_path=RAW_FILE_PATH,
              data_path=DATA_PATH,
              pulse_path=PULSE_PATH,
              train_path=TRAIN_PATH):
    raw_data = RawDataSplit(file_path=file_path.format(module_id),
                            data_path=data_path.format(module_id),
                            train_path=train_path.format(module_id),
                            pulse_path=pulse_path.format(module_id))
    data = raw_data.get_ordered_data(pids=4)
    print("Pulse ID: {0:d}, Data shape: {1}".format(data['pulseId'][0], data['data'].shape))
    gain_level = GainLevel()
    hg_data = gain_level.get_module_data(data, module_id, HIGH_GAIN)
    print("HG_data shape: {}".format(hg_data.shape))
    zero_adu, one_adu = hg_gui_calibrate(hg_data, raw_data.data_file)
    print("Zero ADU: {0:.1f}, One ADU: {1:.1f}".format(zero_adu, one_adu))

def calib_proc(module_id=14,
               file_path=PROC_FILE_PATH,
               data_path=DATA_PATH,
               gain_path=GAIN_PATH,
               pulse_path=PULSE_PATH,
               train_path=TRAIN_PATH):
    raw_data = RawData(file_path=file_path.format(module_id),
                       data_path=data_path.format(module_id),
                       gain_path=gain_path.format(module_id),
                       train_path=train_path.format(module_id),
                       pulse_path=pulse_path.format(module_id))
    data = raw_data.get_ordered_data(pids=4)
    print("Pulse ID: {0:d}, Data shape: {1}".format(data['pulseId'][0], data['data'].shape))
    hg_data = data['data'][data['gain'] == 0]
    print("HG_data shape: {}".format(hg_data.shape))
    zero_adu, one_adu = hg_gui_calibrate(hg_data, raw_data.data_file)
    print("Zero ADU: {0:.1f}, One ADU: {1:.1f}".format(zero_adu, one_adu))

if __name__ == "__main__":
    calib_proc()
