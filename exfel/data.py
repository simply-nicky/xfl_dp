"""
wrapper.py - a module with main data processing class implementations
"""
import argparse
import os
import concurrent.futures
import numpy as np
import h5py
from . import utils

DATA_PATH = "entry_1/instrument_1/detector_1/detector_corrected/data"
TRAIN_PATH = "/instrument/trainID"
PULSE_PATH = "/instrument/pulseID"
RAW_DATA_PATH = "/INSTRUMENT/MID_DET_AGIPD1M-1/DET/{:d}CH0:xtdf/image/data"
RAW_TRAIN_PATH = "/INSTRUMENT/MID_DET_AGIPD1M-1/DET/{:d}CH0:xtdf/image/trainId"
RAW_PULSE_PATH = "/INSTRUMENT/MID_DET_AGIPD1M-1/DET/{:d}CH0:xtdf/image/pulseId"
GAIN_PATH = "/INSTRUMENT/MID_DET_AGIPD1M-1/DET/{:d}CH0:xtdf/image/gain"

class Pool(object):
    def __init__(self, num_workers=utils.CORES_COUNT):
        self.executor = concurrent.futures.ProcessPoolExecutor(max_workers=num_workers)
        self.futures = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        return self.executor.__exit__(exc_type, exc_value, exc_tb)

    def submit(self, func, *args, **kwargs):
        self.futures.append(self.executor.submit(func, *args, **kwargs))

    def shutdown(self, wait=True):
        self.executor.shutdown(wait)

    def get(self, out_dict):
        for fut in self.futures:
            chunk = fut.result()
            for key in chunk:
                out_dict[key].append(chunk[key])
        for key in out_dict:
            out_dict[key] = np.concatenate(out_dict[key])
        return out_dict

class CheetahData(object):
    OUT_FOLDER = os.path.dirname(os.path.dirname(__file__))
    PIDS = 4 * np.arange(0, 176)
    DATA_KEY = utils.DATA_KEY
    PULSE_KEY = utils.PULSE_KEY
    TRAIN_KEY = utils.TRAIN_KEY

    def __init__(self,
                 file_path=utils.BASE_PATH,
                 data_path=DATA_PATH,
                 pulse_path=PULSE_PATH,
                 train_path=TRAIN_PATH):
        self.file_path = file_path
        self.data_path, self.pulse_path, self.train_path = data_path, pulse_path, train_path

    @property
    def out_path(self):
        return os.path.join(self.OUT_FOLDER, "hdf5/{}".format(os.path.basename(self.file_path)))

    @property
    def size(self):
        return self.data.shape[0]

    @property
    def dimensions(self):
        return len(self.data.shape)

    @property
    def data_file(self):
        return h5py.File(self.file_path, 'r')

    @property
    def data(self):
        return self.data_file[self.data_path]

    @property
    def train_ids(self):
        return self.data_file[self.train_path]

    @property
    def pulse_ids(self):
        return self.data_file[self.pulse_path]

    @property
    def chunks(self):
        limits = np.linspace(0, self.size, utils.CORES_COUNT + 1).astype(int)
        return list(zip(limits[:-1], limits[1:]))

    def empty_dict(self):
        return dict([(self.DATA_KEY, []),
                     (self.PULSE_KEY, []),
                     (self.TRAIN_KEY, [])])

    def data_dict(self, data, pulse_ids, train_ids):
        return dict([(self.DATA_KEY, data),
                     (self.PULSE_KEY, pulse_ids),
                     (self.TRAIN_KEY, train_ids)])

    def get_data_chunk(self, start, stop):
        return self.data_dict(data=self.data[start:stop],
                              train_ids=self.train_ids[start:stop],
                              pulse_ids=self.pulse_ids[start:stop])

    def get_data(self):
        pool = Pool()
        with pool:
            for start, stop in self.chunks:
                pool.submit(self.get_data_chunk, start, stop)
        return pool.get(self.empty_dict())

    def get_filtered_data_chunk(self, start, stop, limit):
        data_chunk = self.data[start:stop]
        pids_chunk = self.pulse_ids[start:stop]
        tids_chunk = self.train_ids[start:stop]
        axis = tuple(np.arange(1, self.dimensions))
        idxs = np.where(data_chunk.max(axis=axis) > limit)
        return self.data_dict(data=data_chunk[idxs],
                              pulse_ids=pids_chunk[idxs],
                              train_ids=tids_chunk[idxs])

    def get_filtered_data(self, limit):
        pool = Pool()
        with pool:
            for start, stop in self.chunks:
                pool.submit(self.get_filtered_data_chunk, start, stop, limit)
        return pool.get(self.empty_dict())

    def get_ordered_data_chunk(self, start, stop, pid):
        data_chunk = self.data[start:stop]
        pids_chunk = self.pulse_ids[start:stop]
        tids_chunk = self.train_ids[start:stop]
        idxs = np.where(pids_chunk == pid)
        return self.data_dict(data=data_chunk[idxs],
                              pulse_ids=np.array([pid], dtype=np.uint64),
                              train_ids=tids_chunk[idxs])

    def get_ordered_data(self, pids=None):
        if pids is None:
            _pids = self.PIDS
        elif isinstance(pids, int):
            _pids = [pids]
        else:
            _pids = pids
        results = []
        for pid in _pids:
            pool = Pool()
            with pool:
                for start, stop in self.chunks:
                    pool.submit(self.get_ordered_data_chunk, start, stop, pid)
            res = pool.get(self.empty_dict())
            if res[self.DATA_KEY].any():
                results.append(res)
        if len(results) == 1:
            results = results[0]
        return results

    def _create_out_file(self):
        utils.make_output_dir(os.path.dirname(self.out_path))
        return h5py.File(self.out_path, 'w')

    def _save_parameters(self, out_file):
        arg_group = out_file.create_group('arguments')
        arg_group.create_dataset('file_path', data=np.string_(self.file_path))
        arg_group.create_dataset('data_path', data=self.data_path)
        arg_group.create_dataset('pulseId_path', data=self.pulse_path)
        arg_group.create_dataset('trainId_path', data=self.train_path)

    def _save_data(self, out_file):
        data = self.get_data()
        data_group = out_file.create_group('data')
        for key in data:
            if key == self.DATA_KEY:
                data_group.create_dataset(key, data=data[key], compression='gzip')
            else:
                data_group.create_dataset(key, data=data[key])

    def save(self):
        out_file = self._create_out_file()
        self._save_parameters(out_file)
        self._save_data(out_file)      
        out_file.close()

    def _save_ordered_data(self, out_file, pids=None):
        data_list = self.get_ordered_data(pids)
        data_group = out_file.create_group('data')
        for data in data_list:
            pid_group = data_group.create_group("pulseId {:d}".format(data[self.PULSE_KEY][0]))
            for key in data:
                if key == self.DATA_KEY:
                    pid_group.create_dataset(key, data=data[key], compression='gzip')
                elif key == self.PULSE_KEY:
                    continue
                else:
                    pid_group.create_dataset(key, data=data[key])

class RawData(CheetahData):
    GAIN_KEY = utils.GAIN_KEY

    def __init__(self,
                 file_path=utils.BASE_PATH,
                 data_path=RAW_DATA_PATH,
                 gain_path=GAIN_PATH,
                 pulse_path=RAW_PULSE_PATH,
                 train_path=RAW_TRAIN_PATH):
        super(RawData, self).__init__(file_path,
                                      data_path,
                                      pulse_path,
                                      train_path)
        self.gain_path = gain_path

    @property
    def gain(self):
        return self.data_file[self.gain_path]

    def empty_dict(self):
        return dict([(self.DATA_KEY, []),
                     (self.GAIN_KEY, []),
                     (self.PULSE_KEY, []),
                     (self.TRAIN_KEY, [])])

    def data_dict(self, data, gain, pulse_ids, train_ids):
        return dict([(self.DATA_KEY, data),
                     (self.GAIN_KEY, gain),
                     (self.PULSE_KEY, pulse_ids),
                     (self.TRAIN_KEY, train_ids)])

    def get_data_chunk(self, start, stop):
        return self.data_dict(data=self.data[start:stop],
                              gain=self.gain[start:stop],
                              train_ids=self.train_ids[start:stop],
                              pulse_ids=self.pulse_ids[start:stop])

    def get_filtered_data_chunk(self, start, stop, limit):
        data_chunk = self.data[start:stop]
        gain_chunk = self.gain[start:stop]
        pids_chunk = self.pulse_ids[start:stop]
        tids_chunk = self.train_ids[start:stop]
        axis = tuple(np.arange(1, self.dimensions))
        idxs = np.where(data_chunk.max(axis=axis) > limit)
        return self.data_dict(data=data_chunk[idxs],
                              gain=gain_chunk[idxs],
                              pulse_ids=pids_chunk[idxs],
                              train_ids=tids_chunk[idxs])

    def get_ordered_data_chunk(self, start, stop, pid):
        data_chunk = self.data[start:stop]
        gain_chunk = self.gain[start:stop]
        pids_chunk = self.pulse_ids[start:stop]
        tids_chunk = self.train_ids[start:stop]
        idxs = np.where(pids_chunk == pid)
        return self.data_dict(data=data_chunk[idxs],
                              gain=gain_chunk[idxs],
                              pulse_ids=np.array([pid], dtype=np.uint64),
                              train_ids=tids_chunk[idxs])

    def _save_parameters(self, out_file):
        arg_group = out_file.create_group('arguments')
        arg_group.create_dataset('file_path', data=np.string_(self.file_path))
        arg_group.create_dataset('data_path', data=self.data_path)
        arg_group.create_dataset('gain_path', data=self.gain_path)
        arg_group.create_dataset('pulseId_path', data=self.pulse_path)
        arg_group.create_dataset('trainId_path', data=self.train_path)

class RawDataSplit(CheetahData):
    GAIN_KEY = utils.GAIN_KEY

    def __init__(self,
                 file_path=utils.BASE_PATH,
                 data_path=RAW_DATA_PATH,
                 pulse_path=RAW_PULSE_PATH,
                 train_path=RAW_TRAIN_PATH):
        super(RawDataSplit, self).__init__(file_path=file_path,
                                           data_path=data_path,
                                           pulse_path=pulse_path,
                                           train_path=train_path)

    def empty_dict(self):
        return dict([(self.DATA_KEY, []),
                     (self.GAIN_KEY, []),
                     (self.PULSE_KEY, []),
                     (self.TRAIN_KEY, [])])

    def data_dict(self, data, gain, pulse_ids, train_ids):
        return dict([(self.DATA_KEY, data),
                     (self.GAIN_KEY, gain),
                     (self.PULSE_KEY, pulse_ids),
                     (self.TRAIN_KEY, train_ids)])

    def get_data_chunk(self, start, stop):
        return self.data_dict(data=self.data[start:stop, 0],
                              gain=self.data[start:stop, 1],
                              train_ids=self.train_ids[start:stop],
                              pulse_ids=self.pulse_ids[start:stop])

    def get_filtered_data_chunk(self, start, stop, limit):
        data_chunk = self.data[start:stop, 0]
        gain_chunk = self.data[start:stop, 1]
        pids_chunk = self.pulse_ids[start:stop]
        tids_chunk = self.train_ids[start:stop]
        axis = tuple(np.arange(1, self.dimensions))
        idxs = np.where(data_chunk.max(axis=axis) > limit)
        return self.data_dict(data=data_chunk[idxs],
                              gain=gain_chunk[idxs],
                              pulse_ids=pids_chunk[idxs],
                              train_ids=tids_chunk[idxs])

    def get_ordered_data_chunk(self, start, stop, pid):
        data_chunk = self.data[start:stop]
        gain_chunk = self.data[start:stop]
        pids_chunk = self.pulse_ids[start:stop]
        tids_chunk = self.train_ids[start:stop]
        idxs = np.where(pids_chunk == pid)
        return self.data_dict(data=data_chunk[idxs],
                              gain=gain_chunk[idxs],
                              pulse_ids=np.array([pid], dtype=np.uint64),
                              train_ids=tids_chunk[idxs])

def main():
    parser = argparse.ArgumentParser(description='Run XFEL post processing of cheetah data')
    parser.add_argument('rnum', type=int, help='run number')
    parser.add_argument('cnum', type=int, help='cheetah number')
    parser.add_argument('tag', type=str, help='cheetah tag associated with the current run (written after a hyphen in the cheetah folder name)')
    parser.add_argument('limit', type=int, nargs='?', default=20000, help='minimum ADU value to trim out black images')
    parser.add_argument('outdir', type=str, nargs='?', default=os.path.dirname(os.path.dirname(__file__)), help='output folder location to write processed data')
    parser.add_argument('-off', '--offline', action='store_true', help='offline - run not in Maxwell cluster for debug purposes')
    parser.add_argument('-v', '--verbosity', action='store_true', help='increase output verbosity')
    args = parser.parse_args()

    xfl_data = CheetahData(args.rnum, args.cnum, args.limit)
    if args.verbosity:
        print("List of typed arguments:")
        for key, val in vars(args).items():
            print(key, val, sep=' = ')
        print("cheetah data is located in %s" % xfl_data.file_path)
        print("Writing data to folder: %s" % xfl_data.out_path)
        xfl_data.save()
        print("Done")
    else:
        xfl_data.save()
        