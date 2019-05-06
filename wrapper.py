from __future__ import print_function
from . import xfl_dp, utils
import argparse, os

class XFLData:
    def __init__(self, rnum, cnum, tag, output_folder=os.path.dirname(__file__)):
        self.cheetah_path = utils.get_path_to_data(rnum, cnum, tag, 'cxi', True)
        self.data_size = xfl_dp.get_data_size(rnum, cnum, tag, 'cxi', True)
        self.output_path = utils.output_path(rnum, cnum, 'cxi', output_folder)

    def data(self, limit=500, normalize=True):
        return xfl_dp.data(self.cheetah_path, self.data_size, limit, normalize, True)

    def data_serial(self, limit=500, normalize=True):
        return xfl_dp.data_serial(self.cheetah_path, limit, normalize, True)

    def write(self, limit=500, normalize=True):
        xfl_dp.write_data(self.cheetah_path, self.output_path, self.data_size, limit, normalize, True)

def main():
    parser = argparse.ArgumentParser(description='Run XFEL post processing of cheetah data')
    parser.add_argument('rnum', type=int, help='run number')
    parser.add_argument('cnum', type=int, help='cheetah number')
    parser.add_argument('tag', type=str, help='cheetah tag associated with the current run (written after a hyphen in the cheetah folder name)')
    parser.add_argument('limit', type=int, nargs='?', default=500, help='minimum ADU value to trim out black images')
    parser.add_argument('outpath', type=str, nargs='?', default=os.path.dirname(__file__), help='output folder location to write processed data')
    parser.add_argument('-n', '--normalize', action='store_true', help='normalize frame intensities')
    parser.add_argument('-v', '--verbosity', action='store_true', help='increase output verbosity')
    args = parser.parse_args()

    xfl_data = XFLData(args.rnum, args.cnum, args.tag, args.outpath)
    if args.verbosity:
        print("List of typed arguments:")
        for key, val in vars(args).items():
            print(key, val, sep=' = ')
        print("cheetah data is located in %s" % xfl_data.cheetah_path)
        print("Writing data to folder: %s" % xfl_data.output_path)
        xfl_data.write(args.limit, args.normalize)
        print("Done")
    else:
        xfl_data.write(args.limit, args.normalize)