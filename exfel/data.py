"""
wrapper.py - a module with main data processing class implementations
"""
import argparse
import os
import concurrent.futures
from abc import ABCMeta, abstractmethod
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

class ABCData(metaclass=ABCMeta):
    OUT_FOLDER = os.path.dirname(os.path.dirname(__file__))
    OUT_PATH = os.path.join(OUT_FOLDER, "hdf5/r{0:04d}/XFEL-r{0:04d}-c{1:02d}.h5")
    PIDS = 4 * np.arange(0, 176)

    def __init__(self,
                 rnum=None,
                 cnum=None,
                 file_path=utils.BASE_PATH,
                 data_path=DATA_PATH,
                 pulse_path=PULSE_PATH,
                 train_path=TRAIN_PATH):
        self.rnum, self.cnum = rnum, cnum
        self.file_path = file_path.format(self.rnum, self.cnum)
        self.data_path, self.pulse_path, self.train_path = data_path, pulse_path, train_path

    @property
    def out_path(self):
        return self.OUT_PATH(self.rnum, self.cnum)

    @property
    def size(self):
        return self.data.shape[0]

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

    @abstractmethod
    def get_data_chunk(self, start, stop):
        pass

    def get_data(self):
        ranges = utils.chunkify(0, self.size)
        fut_list = []
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for start, stop in ranges:
                fut_list.append(executor.submit(self.get_data_chunk, start, stop))
        data_list, tids_list, pids_list = [], [], []
        for fut in fut_list:
            data_chunk, tids_chunk, pids_chunk = fut.result()
            data_list.append(data_chunk)
            tids_list.append(tids_chunk)
            pids_list.append(pids_chunk)
        return np.concatenate(data_list), np.concatenate(tids_list), np.concatenate(pids_list)

    def get_ordered_data_chunk(self, start, stop, pid):
        data_chunk, tids_chunk, pids_chunk = self.get_data_chunk(start, stop)
        idxs = np.where(pids_chunk == pid)
        return data_chunk[idxs], tids_chunk[idxs]

    def get_ordered_data(self, pids=None):
        _pids = self.PIDS if pids is None else np.array(pids)
        fut_lists = [[] for _ in _pids]
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for pid, fut_list in zip(_pids, fut_lists):
                for start, stop in utils.chunkify(0,
                                                  self.size,
                                                  utils.CORES_COUNT // len(_pids) + 1):
                    fut_list.append(executor.submit(self.get_ordered_data_chunk, start, stop, pid))
        data_list, tids_list, pids_idxs = [], [], []
        for idx, fut_list in enumerate(fut_lists):
            pid_data_list, pid_tids_list = [], []
            for fut in fut_list:
                data_chunk, tids_chunk = fut.result()
                if data_chunk.any():
                    pid_data_list.append(data_chunk)
                    pid_tids_list.append(tids_chunk)
            if data_list:
                data_list.append(np.concatenate(pid_data_list))
                tids_list.append(np.concatenate(pid_tids_list))
                pids_idxs.append(idx)
        return data_list, tids_list, _pids[pids_idxs]

    def _create_outfile(self):
        utils.make_output_dir(os.path.dirname(self.out_path))
        return h5py.File(self.out_path, 'w')

    def _save_parameters(self, outfile):
        arg_group = outfile.create_group('arguments')
        arg_group.create_dataset('data_file', data=np.string_(self.file_path))
        arg_group.create_dataset('run_number', data=self.rnum)
        arg_group.create_dataset('c_number', data=self.cnum)
        arg_group.create_dataset('data_type', data=self.__class__.__name__)

    def _save_data(self, outfile):
        data, tids, pids = self.get_data()
        data_group = outfile.create_group('data')
        data_group.create_dataset('data', data=data, compression='gzip')
        data_group.create_dataset('trainID', data=tids)
        data_group.create_dataset('pulseID', data=pids)

    def save(self):
        out_file = self._create_outfile()
        self._save_parameters(out_file)
        self._save_data(out_file)      
        out_file.close()

    def _save_ordered_data(self, out_file, pids=None):
        data_list, tids_list = self.get_ordered_data(pids)
        data_group = out_file.create_group('data')
        for pid, data, tids in zip(pids, data_list, tids_list):
            pid_group = data_group.create_group(str(pid))
            pid_group.create_dataset('data', data=data, compression='gzip')
            pid_group.create_dataset('trainID', data=tids)
        data_group.create_dataset('pulseID', data=pids)

class Data(ABCData):
    def get_data_chunk(self, start, stop):
        return (self.data[start:stop],
                self.train_ids[start:stop],
                self.pulse_ids[start:stop])

class TrimData(ABCData):
    def __init__(self,
                 rnum=None,
                 cnum=None,
                 limit=20000,
                 file_path=utils.BASE_PATH,
                 data_path=DATA_PATH,
                 pulse_path=PULSE_PATH,
                 train_path=TRAIN_PATH):
        super(TrimData, self).__init__(rnum, cnum, file_path, data_path, pulse_path, train_path)
        self.limit = limit

    def get_data_chunk(self, start, stop):
        data_chunk = self.data[start:stop]
        pids_chunk = self.pulse_ids[start:stop]
        tids_chunk = self.train_ids[start:stop]
        idxs = np.where(data_chunk.max(axis=(1, 2)) > self.limit)
        return data_chunk[idxs], tids_chunk[idxs], pids_chunk[idxs]

class ABCRawData(ABCData):
    def __init__(self,
                 rnum=None,
                 cnum=None,
                 file_path=utils.BASE_PATH,
                 data_path=RAW_DATA_PATH,
                 gain_path=GAIN_PATH,
                 pulse_path=RAW_PULSE_PATH,
                 train_path=RAW_TRAIN_PATH):
        super(ABCRawData, self).__init__(rnum, cnum, file_path, data_path, pulse_path, train_path)
        self.gain_path = gain_path

    @property
    def gain(self):
        return self.data_file[self.gain_path]

    def get_data(self):
        ranges = utils.chunkify(0, self.size)
        fut_list = []
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for start, stop in ranges:
                fut_list.append(executor.submit(self.get_data_chunk, start, stop))
        data_list, gain_list, tids_list, pids_list = [], [], [], []
        for fut in fut_list:
            data_chunk, gain_chunk, tids_chunk, pids_chunk = fut.result()
            data_list.append(data_chunk)
            gain_list.append(gain_chunk)
            tids_list.append(tids_chunk)
            pids_list.append(pids_chunk)
        return np.concatenate(data_list), np.concatenate(gain_list), np.concatenate(tids_list), np.concatenate(pids_list)

    def get_ordered_data_chunk(self, start, stop, pid):
        data_chunk, gain_chunk, tids_chunk, pids_chunk = self.get_data_chunk(start, stop)
        idxs = np.where(pids_chunk == pid)
        return data_chunk[idxs], gain_chunk[idxs], tids_chunk[idxs]

    def get_ordered_data(self, pids=None):
        _pids = self.PIDS if pids is None else np.array(pids)
        fut_lists = [[] for _ in _pids]
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for pid, fut_list in zip(_pids, fut_lists):
                for start, stop in utils.chunkify(0,
                                                  self.size,
                                                  utils.CORES_COUNT // len(_pids) + 1):
                    fut_list.append(executor.submit(self.get_ordered_data_chunk, start, stop, pid))
        data_list, gain_list, tids_list, pids_idxs = [], [], [], []
        for idx, fut_list in enumerate(fut_lists):
            pid_data_list, pid_gain_list, pid_tids_list = [], [], []
            for fut in fut_list:
                data_chunk, gain_chunk, tids_chunk = fut.result()
                if data_chunk.any():
                    pid_data_list.append(data_chunk)
                    pid_gain_list.append(gain_chunk)
                    pid_tids_list.append(tids_chunk)
            print(len(pid_data_list))
            if pid_data_list:
                data_list.append(np.concatenate(pid_data_list))
                gain_list.append(np.concatenate(pid_gain_list))
                tids_list.append(np.concatenate(pid_tids_list))
                pids_idxs.append(idx)
        return data_list, gain_list, tids_list, _pids[pids_idxs]

    def _save_data(self, outfile):
        data, gain, tids, pids = self.get_data()
        data_group = outfile.create_group('data')
        data_group.create_dataset('data', data=data, compression='gzip')
        data_group.create_dataset('gain', data=gain, compression='gzip')
        data_group.create_dataset('trainID', data=tids)
        data_group.create_dataset('pulseID', data=pids)

    def _save_ordered_data(self, out_file, pids=None):
        data_list, gain_list, tids_list = self.get_ordered_data(pids)
        data_group = out_file.create_group('data')
        for pid, data, gain, tids in zip(pids, data_list, gain_list, tids_list):
            pid_group = data_group.create_group(str(pid))
            pid_group.create_dataset('data', data=data, compression='gzip')
            pid_group.create_dataset('gain', data=gain, compression='gzip')
            pid_group.create_dataset('trainID', data=tids)
        data_group.create_dataset('pulseID', data=pids)

class RawData(ABCRawData):
    def get_data_chunk(self, start, stop):
        return (self.data[start:stop],
                self.gain[start:stop],
                self.train_ids[start:stop],
                self.pulse_ids[start:stop])

class RawTrimData(ABCRawData):
    def __init__(self,
                 rnum=None,
                 cnum=None,
                 limit=20000,
                 file_path=utils.BASE_PATH,
                 data_path=RAW_DATA_PATH,
                 gain_path=GAIN_PATH,
                 pulse_path=RAW_PULSE_PATH,
                 train_path=RAW_TRAIN_PATH):
        super(RawTrimData, self).__init__(rnum, cnum, file_path, data_path, gain_path, pulse_path, train_path)
        self.limit = limit

    def get_data_chunk(self, start, stop):
        data_chunk = self.data[start:stop]
        gain_chunk = self.gain[start:stop]
        pids_chunk = self.pulse_ids[start:stop]
        tids_chunk = self.train_ids[start:stop]
        idxs = np.where(data_chunk.max(axis=(1, 2)) > self.limit)
        return data_chunk[idxs], gain_chunk[idxs], tids_chunk[idxs], pids_chunk[idxs]

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

    xfl_data = TrimData(args.rnum, args.cnum, args.limit)
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
        