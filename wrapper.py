from __future__ import print_function
from .src import utils, data, data_mpi, splitted_data, write_data, write_mpi
import argparse, os, numpy as np
from multiprocessing import cpu_count

class Scan(object):
    def __init__(self, rnum, cnum, tag, output_folder=os.path.dirname(os.path.dirname(__file__)), online=True):
        self.cheetah_path = utils.get_path_to_data(rnum, cnum, tag, 'cxi', online)
        self.output_path = utils.output_path(rnum, cnum, 'cxi', output_folder)
        self.data_size = utils.get_data_size(self.cheetah_path)

    def data(self, limit=20000):
        return data(self.cheetah_path, self.data_size, limit)

    def data_mpi(self, n_procs=cpu_count(), limit=20000):
       data_list, tids_list, pids_list = data_mpi(self.cheetah_path, self.data_size, n_procs, limit)
       return np.concatenate(data_list), np.concatenate(tids_list), np.concatenate(pids_list)

    def splitted_data(self, pids, limit=20000):
        return splitted_data(self.cheetah_path, self.data_size, list(pids), limit)

    def write(self, limit=20000):
        write_data(self.cheetah_path, self.output_path, self.data_size, limit)

    def write_mpi(self, n_procs=cpu_count(), limit=20000):
        write_mpi(self.cheetah_path, self.output_path, self.data_size, n_procs, limit)

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

    xfl_data = Scan(args.rnum, args.cnum, args.tag, args.outdir, not args.offline)
    if args.verbosity:
        print("List of typed arguments:")
        for key, val in vars(args).items():
            print(key, val, sep=' = ')
        print("cheetah data is located in %s" % xfl_data.cheetah_path)
        print("Writing data to folder: %s" % xfl_data.output_path)
        xfl_data.write(args.limit)
        print("Done")
    else:
        xfl_data.write(args.limit)