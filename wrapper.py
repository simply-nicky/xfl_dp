import argparse
import os
import concurrent.futures
from abc import ABCMeta, abstractmethod
import numpy as np
import h5py
from . import utils

class DataFactory(object):
    def __init__(self,
                 file_path=utils.BASE_PATH,
                 data_path=utils.DATA_PATH,
                 pulse_path=utils.PULSE_PATH,
                 train_path=utils.TRAIN_PATH):
        self.file_path, self.data_path = file_path, data_path
        self.pulse_path, self.train_path = pulse_path, train_path

    def __call__(self, Class, *args, **kwargs):
        obj = Class(*args, **kwargs)
        obj.FILE_PATH = self.file_path
        obj.DATA_PATH = self.data_path
        obj.PULSE_PATH = self.pulse_path
        obj.TRAIN_PATH = self.train_path
        return obj

class ABCData(metaclass=ABCMeta):
    FILE_PATH = utils.BASE_PATH
    OUT_FOLDER = os.path.dirname(os.path.dirname(__file__))
    OUT_PATH = os.path.join(OUT_FOLDER, "hdf5/r{0:04d}/XFEL-r{0:04d}-c{1:02d}.h5")
    DATA_PATH = utils.DATA_PATH
    PULSE_PATH = utils.PULSE_PATH
    TRAIN_PATH = utils.TRAIN_PATH

    def __init__(self, rnum=None, cnum=None):
        self.rnum, self.cnum = rnum, cnum

    @property
    def file_path(self):
        return self.FILE_PATH.format(self.rnum, self.cnum)

    @property
    def out_path(self):
        return self.OUT_PATH(self.rnum, self.cnum)

    @property
    def size(self):
        return self.data.shape[0]

    @property
    def raw_file(self):
        return h5py.File(self.file_path, 'r')

    @property
    def data(self):
        return self.raw_file[self.DATA_PATH]

    @property
    def train_ids(self):
        return self.raw_file[self.TRAIN_PATH]

    @property
    def pulse_ids(self):
        return self.raw_file[self.PULSE_PATH]

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

    def get_ordered_data(self, pids):
        pids_list = list(pids)
        fut_lists = [[] for _ in pids_list]
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for pid, fut_list in zip(pids_list, fut_lists):
                for start, stop in utils.chunkify(0,
                                                  self.size,
                                                  utils.CORES_COUNT // len(pids_list) + 1):
                    fut_list.append(executor.submit(self.get_ordered_data_chunk, start, stop, pid))
        data_list, tids_list = [], []
        for fut_list in fut_lists:
            pid_data_list, pid_tids_list = [], []
            for fut in fut_list:
                data_chunk, tids_chunk = fut.result()
                if data_chunk.any():
                    pid_data_list.append(data_chunk)
                    pid_tids_list.append(tids_chunk)
            data_list.append(np.concatenate(pid_data_list))
            tids_list.append(np.concatenate(pid_tids_list))
        return data_list, tids_list

    def _create_outfile(self):
        utils.make_output_dir(os.path.dirname(self.out_path))
        return h5py.File(self.out_path, 'w')

    def _save_parameters(self, outfile):
        arg_group = outfile.create_group('arguments')
        arg_group.create_dataset('data_path', data=np.string_(self.file_path))
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

    def _save_ordered_data(self, out_file, pids):
        data_list, tids_list = self.get_ordered_data(pids)
        data_group = out_file.create_group('data')
        for pid, data, tids in zip(pids, data_list, tids_list):
            pid_group = data_group.create_group(str(pid))
            pid_group.create_dataset('data', data=data, compression='gzip')
            pid_group.create_dataset('trainID', data=tids)
        data_group.create_dataset('pulseID', data=pids)

class Data(ABCData):
    def get_data_chunk(self, start, stop):
        return self.data[start, stop], self.train_ids[start, stop], self.pulse_ids[start, stop]

class TrimData(ABCData):
    def __init__(self, rnum=None, cnum=None, limit=20000):
        super(TrimData, self).__init__(rnum, cnum)
        self.limit = limit

    def get_data_chunk(self, start, stop):
        data_chunk = self.data[start, stop]
        pids_chunk = self.pulse_ids[start, stop]
        tids_chunk = self.train_ids[start, stop]
        idxs = np.where(data_chunk.max(axis=(1, 2)) > self.limit)
        return data_chunk[idxs], tids_chunk[idxs], pids_chunk[idxs]

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
        