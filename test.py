from exfel import RawData

RAW_PATH = "raw/r0221_raw.h5"
RAW_DATA_PATH = "data/ADU"
RAW_GAIN_PATH = "data/DigitalGains"
RAW_PULSE_PATH = "data/pulseID"
RAW_TRAIN_PATH = "data/trainID"

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
