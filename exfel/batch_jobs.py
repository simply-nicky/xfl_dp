import subprocess
import configparser
import argparse
import os

SHELL_SCRIPT = os.path.abspath('process.sh')
CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config.ini')
PACKAGE_NAME = 'exfel'

class ConfigParser(object):
    def __init__(self, config_file):
        self.config_file = os.path.abspath(config_file)
        self.config = configparser.ConfigParser()
        self.config.read(self.config_file)
        self._init_data()

    def _init_data(self):
        self.beam_line = self.config.get('raw_data', 'beam_line')
        self.raw_path = self.config.get('raw_data', 'raw_path')
        self.out_base = self.config.get('raw_data', 'output_path')
        self.dark_base = self.config.get('dark', 'dark_path')
        self.hg_run = self.config.getint('dark', 'hg_run')
        self.mg_run = self.config.getint('dark', 'mg_run')
        self.lg_run = self.config.getint('dark', 'lg_run')

class JobsParser(object):
    BATCH_CMD = 'sbatch'

    def __init__(self, shell_script, config_file):
        self.shell_script = shell_script
        self.config = ConfigParser(config_file)
        self.batch_out = os.path.join(self.config.out_base, 'sbatch_out')

    def batch(self, run_type, **kwparams):
        new_job = Job(self, run_type, **kwparams)
        print('Submitting job {}'.format(new_job.job_name))
        print('Shell script: {}'.format(self.shell_script))
        print('Command: {}'.format(' '.join(new_job.cmd)))
        output = subprocess.call(new_job.cmd, check=True, capture_output=True)
        job_num = output.stdout.rstrip().decode("unicode_escape")
        print("The job {0:s} has been submitted".format(new_job.job_name))
        print("Job ID: {:d}".format(job_num))
        return job_num

class Job(object):
    JOB_NAME = {'list': "list_r{run_number:04d}",
                'pid': "pid_r{run_number:04d}_pid{pid:02d}_AGIPD{module_id:2d}",
                'hg': "hg_r{run_number:04d}_pid{pid:02d}_AGIPD{module_id:2d}"}

    def __init__(self, jobs_parser, run_type, **kwparams):
        self.job_parser = jobs_parser
        self.run_type, self.kwparams = run_type, kwparams

    @property
    def job_name(self):
        return self.JOB_NAME[self.run_type].format(self.kwparams)

    @property
    def sbatch_params(self):
        return ['--job_name', self.job_name,
                '--output', os.path.join(self.job_parser.batch_out, '{}.out'.format(self.job_name)),
                '--error', os.path.join(self.job_parser.batch_out, '{}.err'.format(self.job_name))]

    @property
    def shell_params(self):
        params = [os.path.dirname(__file__), PACKAGE_NAME]
        params += [self.kwparams['run_number'], self.run_type]
        params += ['--config_file', self.job_parser.config.config_file]
        if self.run_type in ['hg', 'pid']:
            try:
                params += ['--chunk_number', self.kwparams['chunk_number']]
                params += ['--module_id', self.kwparams['module_id']]
                params += ['--pulse_id', self.kwparams['pulse_id']]
            except KeyError as error:
                error_text = 'Wrong script shell parameters:\n{}'.format(self.kwparams)
                raise ValueError(error_text) from error
        return params

    @property
    def cmd(self):
        cmd = [self.job_parser.BATCH_CMD]
        cmd.extend(self.sbatch_params)
        cmd.append(self.job_parser.shell_script)
        cmd.extend(self.shell_params)
        return cmd

def process_file(file_path):
    filename = os.path.basename(file_path)
    try:
        run_number = filename[filename.index('AGIPD'):filename.index('-S')]
        chunk_number = filename[filename.index('-S'):filename.index('.h5')]
    except ValueError as error:
        raise ValueError('Wrong filename: {}'.format(filename)) from error
    return run_number, chunk_number

def main():
    parser = argparse.ArgumentParser(description='Batch jobs to Maxwell to process AGIPD data')
    parser.add_argument('run_number', type=int, help='run number')
    parser.add_argument('run_type', type=str, choices=['pid', 'hg', 'list'], help='Process type')
    parser.add_argument('--config_file', type=str, default=CONFIG_PATH, help='Configuration file')
    parser.add_argument('--pulse_id', type=int, help='PulseID to extract data')
    args = parser.parse_args()

    jobs_parser = JobsParser(SHELL_SCRIPT, args.config_file)
    jobs_parser.batch(run_type=args.run_type,
                      run_number=args.run_number,
                      config_file=args.config_file)

if __name__ == "__main__":
    main()