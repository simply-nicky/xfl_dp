from __future__ import print_function
from .src import utils, data_mpi, write_mpi
import argparse, os, numpy as np, concurrent.futures, h5py
from multiprocessing import cpu_count
from abc import ABCMeta, abstractproperty, abstractmethod

class ABCData(metaclass=ABCMeta):
    @abstractproperty
    def rnum(self): pass

    @abstractproperty
    def cnum(self): pass

    @abstractproperty
    def tag(self): pass

    @abstractproperty
    def output_folder(self): pass

    @abstractproperty
    def limit(self): pass

    @abstractproperty
    def online(self): pass

    @abstractmethod
    def process_frame(self, frame): pass

    @property
    def cheetah_path(self): return utils.get_path_to_data(self.rnum, self.cnum, self.tag, self.online)

    @property
    def output_path(self): return utils.output_path(self.rnum, self.cnum, self.output_folder)

    @property
    def data_size(self): return utils.get_data_size(self.cheetah_path)

    @property
    def raw_file(self): return h5py.File(self.cheetah_path, 'r')

    @property
    def raw_data(self): return self.raw_file[utils.datapath]

    @property
    def raw_tids(self): return self.raw_file[utils.trainpath]

    @property
    def raw_pids(self): return self.raw_file[utils.pulsepath]

    def data_chunk(self, start, stop):
        data, tidslist, pidslist = [], [], []
        for idx in range(start, stop):
            if self.raw_data[idx].max() > self.limit:
                pidslist.append(self.raw_pids[idx])
                tidslist.append(self.raw_tids[idx])
                data.append(self.process_frame(idx))
        return np.array(data), np.array(tidslist), np.array(pidslist)

    def data(self):
        ranges = utils.chunkify(0, self.data_size)
        datalist, tidslist, pidslist = [], [], []
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for data, tids, pids in executor.map(utils.worker_star(self.data_chunk), ranges):
                if data.any():
                    datalist.append(data)
                    tidslist.append(tids)
                    pidslist.append(pids)
        return np.concatenate(datalist), np.concatenate(tidslist), np.concatenate(pidslist)

    def pid_data_chunk(self, start, stop, pid):
        datalist, tidslist = [], []
        for idx in range(start, stop):
            if self.raw_data[idx].max() > self.limit and self.raw_pids[idx] == pid:
                datalist.append(self.process_frame(idx))
                tidslist.append(self.raw_tids[idx])
        return pid, np.array(datalist), np.array(tidslist)

    def pids_data(self, pids):
        _pids = list(pids)
        datalist, tidslist = [[] for _ in range(len(_pids))], [[] for _ in range(len(_pids))]
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for pid, data, tids in executor.map(utils.worker_star(self.pid_data_chunk), utils.splitted_chunkify(0, self.data_size, _pids)):
                idx = _pids.index(pid)
                if data.any(): datalist[idx].append(data)
                if tids.any(): tidslist[idx].append(tids)
        return [np.concatenate(pid_data) for pid_data in datalist], [np.concatenate(pid_tids) for pid_tids in tidslist]

    def first_frame(self):
        idx = 0
        while self.raw_data[idx].max() < self.limit: idx += 1
        else:
            pid, tid = self.raw_pids[idx], self.raw_tids[idx]
            data = self.process_frame(idx)
        return data[np.newaxis], np.array([tid]), np.array([pid]), idx + 1

    def _save_init(self):
        utils.make_dirs(self.output_path)
        self.outfile = h5py.File(self.output_path, 'w')
        arggroup = self.outfile.create_group('arguments')
        arggroup.create_dataset('cheetah path', data=np.string_(self.cheetah_path))
        arggroup.create_dataset('trimming limit', data=self.limit)
        arggroup.create_dataset('run number', data=self.rnum)
        arggroup.create_dataset('data type', data=self.__class__.__name__)

    def save(self):
        data, tids, pids = self.data()
        self._save_init()
        datagroup = self.outfile.create_group('data')
        datagroup.create_dataset('data', data=data, compression='gzip')
        datagroup.create_dataset('trainID', data=tids)
        datagroup.create_dataset('pulseID', data=pids)        
        self.outfile.close()

    def pids_save(self, pids):
        datalist, tidslist = self.pids_data(pids)
        self._save_init()
        datagroup = self.outfile.create_group('data')
        dgroup = datagroup.create_group('data')
        tidsgroup = datagroup.create_group('traindID')
        for pid, data, tids in zip(pids, datalist, tidslist):
            dgroup.create_dataset(str(pid), data=data, compression='gzip')
            tidsgroup.create_dataset(str(pid), data=tids)
        datagroup.create_dataset('pulseID', data=pids)
        self.outfile.close()

class Data(ABCData):
    rnum, cnum, tag, output_folder, limit, online = None, None, None, None, None, None

    def __init__(self, rnum, cnum, tag, limit=20000, output_folder=os.path.dirname(os.path.dirname(__file__)), online=True):
        self.rnum, self.cnum, self.tag, self.output_folder, self.limit, self.online = rnum, cnum, tag, output_folder, limit, online

    def process_frame(self, idx):
        return utils.apply_agipd_geom(self.raw_data[idx]).astype(np.int32)

    def data_mpi(self, n_procs=cpu_count()):
       data_list, tids_list, pids_list = data_mpi(self.cheetah_path, self.data_size, n_procs, self.limit)
       return np.concatenate(data_list), np.concatenate(tids_list), np.concatenate(pids_list)

    def save_mpi(self, n_procs=cpu_count()):
        write_mpi(self.cheetah_path, self.output_path, self.data_size, n_procs, self.limit)

class PupilData(ABCData):
    rnum, cnum, tag, output_folder, limit, online = None, None, None, None, None, None

    def __init__(self, rnum, cnum, tag, limit=20000, output_folder=os.path.dirname(os.path.dirname(__file__)), online=True):
        self.rnum, self.cnum, self.tag, self.output_folder, self.limit, self.online = rnum, cnum, tag, output_folder, limit, online

    def process_frame(self, idx):
        return utils.apply_agipd_geom(self.raw_data[idx]).astype(np.int32)[utils.pupil_roi]

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

    xfl_data = Data(args.rnum, args.cnum, args.tag, args.limit, args.outdir, not args.offline)
    if args.verbosity:
        print("List of typed arguments:")
        for key, val in vars(args).items():
            print(key, val, sep=' = ')
        print("cheetah data is located in %s" % xfl_data.cheetah_path)
        print("Writing data to folder: %s" % xfl_data.output_path)
        xfl_data.save()
        print("Done")
    else:
        xfl_data.save()