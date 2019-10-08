from exfel import RawData

FILE_PATH = "/gpfs/exfel/exp/MID/201802/p002200/proc/r0221/CORR-R0221-AGIPD00-S00000.h5"
DATA_PATH = "/INSTRUMENT/MID_DET_AGIPD1M-1/DET/0CH0:xtdf/image/data"
GAIN_PATH = "/INSTRUMENT/MID_DET_AGIPD1M-1/DET/0CH0:xtdf/image/gain"
TRAIN_PATH = "/INSTRUMENT/MID_DET_AGIPD1M-1/DET/0CH0:xtdf/image/trainId"
PULSE_PATH = "/INSTRUMENT/MID_DET_AGIPD1M-1/DET/0CH0:xtdf/image/pulseId"

def main(file_path=FILE_PATH,
         data_path=DATA_PATH,
         gain_path=GAIN_PATH,
         pulse_path=PULSE_PATH,
         train_path=TRAIN_PATH):
    data = RawData(file_path=file_path,
                   data_path=data_path,
                   gain_path=gain_path,
                   train_path=train_path,
                   pulse_path=pulse_path)
    data = data.get_ordered_data(pids=4)
    

if __name__ == "__main__":
    main()
