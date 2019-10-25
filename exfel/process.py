import os
import argparse
from .data import RawModuleJoined
from .calib import DarkAGIPD, CalibData
from .batch_jobs import ConfigParser

CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.ini')
BEAM_LINES = ('DETLAB', 'FXE', 'HED', 'HSLAB', 'ITLAB', 'LA1',
              'LA2', 'MID', 'SA1', 'SA2', 'SA3', 'SCS', 'SPB',
              'SQS', 'TST', 'TSTSA2', 'TSTSA3', 'XMPL')

class Process(object):
    DATA_STRUCTURE = "raw/r{run_number:04d}/RAW-R{run_number:04d}-AGIPD{module_id:02d}-S{chunk_num:05d}.h5"
    DATA_FOLDER = "raw/r{run_number:04d}"
    OUT_PID_PATH = "r{run_number:04d}/AGIPD{module_id:02d}-{tag:s}{pid:03d}.h5"
    DARK_CALIB_PATH = "r{hg_run:04d}-r{mg_run:04d}-r{lg_run:04d}/Cheetah-AGIPD-calib.h5"
    DATA_PATH = "/INSTRUMENT/{beam_line:s}_DET_AGIPD1M-1/DET/{module_id:d}CH0:xtdf/image/data"
    TRAIN_PATH = "/INSTRUMENT/{beam_line:s}_DET_AGIPD1M-1/DET/{module_id:d}CH0:xtdf/image/trainId"
    PULSE_PATH = "/INSTRUMENT/{beam_line:s}_DET_AGIPD1M-1/DET/{module_id:d}CH0:xtdf/image/pulseId"

    def __init__(self, run_number, config_file='config.ini'):
        self.run_number = run_number
        self.config = ConfigParser(config_file)
        self._init_paths()
        self._init_dark()

    def _init_paths(self):
        if self.config.beam_line not in BEAM_LINES:
            raise ValueError('Wrong beam_line value: {}'.format(self.config.beam_line))
        if not os.path.exists(self.config.raw_path):
            raise ValueError('Error: can not find the file: {:s}'.format(self.config.raw_path))
        os.makedirs(self.config.out_base, exist_ok=True)

    def _init_dark(self):
        dark_path = os.path.join(self.config.dark_base,
                                 self.DARK_CALIB_PATH.format(hg_run=self.config.hg_run,
                                                             mg_run=self.config.mg_run,
                                                             lg_run=self.config.lg_run))
        if not os.path.exists(dark_path):
            raise ValueError('Wrong dark calibration path: {}'.format(dark_path))
        self.dark_calib = DarkAGIPD(dark_path)

    @property
    def file_folder(self):
        return os.path.join(self.config.raw_path,
                            self.DATA_FOLDER.format(run_number=self.run_number))

    def out_path(self, module_id, pid, tag):
        return os.path.join(self.config.out_base,
                            self.OUT_PID_PATH.format(run_number=self.run_number,
                                                     module_id=module_id,
                                                     pid=pid,
                                                     tag=tag))

    def file_path(self, module_id, chunk_num):
        return os.path.join(self.config.raw_path,
                            self.DATA_STRUCTURE.format(run_number=self.run_number,
                                                       module_id=module_id,
                                                       chunk_num=chunk_num))

    def data_path(self, module_id):
        return self.DATA_PATH.format(self.config.beam_line, module_id)

    def train_path(self, module_id):
        return self.TRAIN_PATH.format(self.config.beam_line, module_id)

    def pulse_path(self, module_id):
        return self.PULSE_PATH.format(self.config.beam_line, module_id)

    def data_file(self, module_id, chunk_num):
        return RawModuleJoined(module_id=module_id,
                               file_path=self.file_path(module_id, chunk_num),
                               data_path=self.data_path(module_id),
                               train_path=self.train_path(module_id),
                               pulse_path=self.pulse_path(module_id))

    def list_files(self):
        return [filename
                for filename in os.listdir(self.file_folder)
                if "AGIPD" in filename]

    def save_cell_data(self, module_id, chunk_num, pid):
        raw_data = self.data_file(module_id, chunk_num)
        out_path = self.out_path(module_id, pid, 'PID')
        print('Reading file: {:s}'.format(raw_data.file_path))
        print('PulseID: {:s}'.format(pid))
        print('Writing to file: {}'.format(out_path))
        raw_data.save_ordered(out_path, pid)

    def save_hg_data(self, module_id, chunk_num, pid):
        raw_data = self.data_file(module_id, chunk_num)
        print('Reading file: {:s}'.format(raw_data.file_path))
        print('PulseID: {:s}'.format(pid))
        data = raw_data.get_ordered_data(pids=pid)
        print('Data shape: {}'.format(data['data'].shape))
        print('Applying dark calibration files: {}'.format(self.dark_calib.data_file.filename))
        dark_module = self.dark_calib.dark_module(module_id)
        calib_data = CalibData(data, dark_module)
        out_path = self.out_path(module_id, pid, 'HG')
        print('Getting the High Gain data')
        print('Writing to file: {}'.format(out_path))
        calib_data.save_hg_data(out_path)

def main():
    parser = argparse.ArgumentParser(description='Run raw AGIPD data processing')
    parser.add_argument('run_number', type=int, help='run number')
    parser.add_argument('run_type', type=str, choices=['pid', 'hg', 'list'], help='Process type')
    parser.add_argument('--config_file', type=str, default=CONFIG_PATH, help='Configuration file')
    parser.add_argument('--chunk_number', type=int, help='chunk number')
    parser.add_argument('--module_id', type=int, help='AGIPD module number')
    parser.add_argument('--pulse_id', type=int, help='PulseID to extract data')
    args = parser.parse_args()

    process = Process(args.run_number, args.config_file)
    if args.run_type == 'pid':
        process.save_cell_data(module_id=args.module_id,
                               chunk_num=args.chunk_number,
                               pid=args.pulse_id)
    elif args.run_type == 'hg':
        process.save_hg_data(module_id=args.module_id,
                             chunk_num=args.chunk_number,
                             pid=args.pulse_id)
    elif args.run_type == 'list':
        files = process.list_files()
        print('\n'.join(files))
    else:
        raise ValueError('Wrong run_type: {}'.format(args.run_type))
    