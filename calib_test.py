from exfel import RawModuleJoined, RawData, DarkCalib, CalibData

RAW_FILE_PATH = "/gpfs/petra3/scratch/alireza2/r0099/RAW-R0099-AGIPD{:02d}-S00003.h5"
DARK_CALIB_PATH = "/gpfs/petra3/scratch/alireza2/scripts/r0096-r0097-r0098/Cheetah-AGIPD{:02d}-calib.h5"
PROC_FILE_PATH = "/gpfs/exfel/exp/MID/201802/p002200/proc/r0283/CORR-R0283-AGIPD{:02d}-S00000.h5"
DATA_PATH = "/INSTRUMENT/SPB_DET_AGIPD1M-1/DET/{:d}CH0:xtdf/image/data"
GAIN_PATH = "/INSTRUMENT/SPB_DET_AGIPD1M-1/DET/{:d}CH0:xtdf/image/gain"
TRAIN_PATH = "/INSTRUMENT/SPB_DET_AGIPD1M-1/DET/{:d}CH0:xtdf/image/trainId"
PULSE_PATH = "/INSTRUMENT/SPB_DET_AGIPD1M-1/DET/{:d}CH0:xtdf/image/pulseId"

def calib_raw(module_id=0,
              file_path=RAW_FILE_PATH,
              calib_path=DARK_CALIB_PATH,
              data_path=DATA_PATH,
              pulse_path=PULSE_PATH,
              train_path=TRAIN_PATH):
    raw_data = RawModuleJoined(module_id=module_id,
                               file_path=file_path,
                               data_path=data_path,
                               train_path=train_path,
                               pulse_path=pulse_path)
    data = raw_data.get_ordered_data(pids=4)
    print("Pulse ID: {0:d}, Data shape: {1}".format(data['pulseId'][0], data['data'].shape))
    dark_calib = DarkCalib(calib_path)
    calib_data = CalibData(data, dark_calib)
    hg_data = calib_data.hg_data()
    print("HG_data shape: {}".format(hg_data.data.shape))
    # zero_adu, one_adu = hg_gui_calibrate(hg_data, raw_data.data_file)
    # print("Zero ADU: {0:.1f}, One ADU: {1:.1f}".format(zero_adu, one_adu))

def proc_save(module_id=14,
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
    raw_data.save_ordered(pids=4)
    print("File saved at location: {}".format(raw_data.out_path))

if __name__ == "__main__":
    calib_raw()
