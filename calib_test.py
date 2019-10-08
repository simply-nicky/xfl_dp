from exfel import RawDataSplit, GainLevel
from exfel.utils import HIGH_GAIN

FILE_PATH = "/gpfs/exfel/exp/MID/201802/p002200/raw/r0221/RAW-R0221-AGIPD{:02d}-S00000.h5"
DATA_PATH = "/INSTRUMENT/MID_DET_AGIPD1M-1/DET/{:d}CH0:xtdf/image/data"
TRAIN_PATH = "/INSTRUMENT/MID_DET_AGIPD1M-1/DET/{:d}CH0:xtdf/image/trainId"
PULSE_PATH = "/INSTRUMENT/MID_DET_AGIPD1M-1/DET/{:d}CH0:xtdf/image/pulseId"

def main(module_id=0,
         file_path=FILE_PATH,
         data_path=DATA_PATH,
         pulse_path=PULSE_PATH,
         train_path=TRAIN_PATH):
    data = RawDataSplit(file_path=file_path.format(module_id),
                        data_path=data_path.format(module_id),
                        train_path=train_path.format(module_id),
                        pulse_path=pulse_path.format(module_id))
    data = data.get_ordered_data(pids=4)
    print("Pulse ID: {0:d}, Data shape: {1}".format(data['pulseId'][0], data['data'].shape))
    gain_level = GainLevel()
    hg_data = gain_level.get_module_data(data, module_id, HIGH_GAIN)
    print("HG_data shape: {}".format(hg_data.shape))

if __name__ == "__main__":
    main()
