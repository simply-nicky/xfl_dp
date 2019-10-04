from exfel import RawData

FILE_PATH = "raw/test_file.h5"
DATA_PATH = "/MID_DET_AGIPD1M-1/CAL/APPEND_RAW/image.data"
GAIN_PATH = "/MID_DET_AGIPD1M-1/CAL/APPEND_RAW/image.gain"
PULSE_PATH = "/MID_DET_AGIPD1M-1/CAL/APPEND_RAW/image.pulseId"
TRAIN_PATH = "/MID_DET_AGIPD1M-1/CAL/APPEND_RAW/image.trainId"

def main(file_path=FILE_PATH,
         data_path=DATA_PATH,
         gain_path=GAIN_PATH,
         pulse_path=PULSE_PATH,
         train_path=TRAIN_PATH):
    data = RawData(file_path=file_path,
                   data_path=data_path,
                   gain_path=gain_path,
                   pulse_path=pulse_path,
                   train_path=train_path)
    print("Size: {:d}".format(data.size))
    print("Gain size: {}".format(data.gain.shape))
    print("Pulse ID size: {}".format(data.data.shape))

if __name__ == "__main__":
    main()
