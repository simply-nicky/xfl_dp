from exfel import RawTrimData

FILE_PATH = "/gpfs/exfel/exp/MID/201802/p002200/proc/r{0:04d}/CORR-R{0:04d}-AGIPD00-S{1:05d}.h5"
DATA_PATH = "/INSTRUMENT/MID_DET_AGIPD1M-1/DET/0CH0:xtdf/image/data"
GAIN_PATH = "/INSTRUMENT/MID_DET_AGIPD1M-1/DET/0CH0:xtdf/image/gain"
TRAIN_PATH = "/INSTRUMENT/MID_DET_AGIPD1M-1/DET/0CH0:xtdf/image/trainId"
PULSE_PATH = "/INSTRUMENT/MID_DET_AGIPD1M-1/DET/0CH0:xtdf/image/pulseId"

def main(file_path=FILE_PATH,
         data_path=DATA_PATH,
         gain_path=GAIN_PATH,
         pulse_path=PULSE_PATH,
         train_path=TRAIN_PATH):
    data = RawTrimData(rnum=221,
                       cnum=0,
                       limit=20000,
                       file_path=file_path,
                       data_path=data_path,
                       gain_path=gain_path,
                       train_path=train_path,
                       pulse_path=pulse_path)
    data_list, _, _, pids = data.get_ordered_data(pids=[0, 4, 8, 12, 16])
    print("Pulse IDs: {0}, data length: {1}".format(pids, len(data_list)))
    for pid, data in zip(pids, data_list):
        print("Pulse ID: {0:d}, data size: {1}".format(pid, data.shape))

if __name__ == "__main__":
    main()
