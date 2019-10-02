import os
import time
import errno
import sys
import h5py
from multiprocessing import cpu_count
import numpy as np
from cfelpyutils.crystfel_utils import load_crystfel_geometry
from cfelpyutils.geometry_utils import apply_geometry_to_data
from mpi4py import MPI

BASE_PATH = "/gpfs/exfel/u/scratch/MID/201802/p002200/cheetah/hdf5/r{0:04d}-data/XFEL-r{0:04d}-c{1:02d}.h5"
USER_PATH = os.path.join(os.path.dirname(__file__), "../../cheetah/XFEL-r{0:04d}-c{1:02d}.cxi")
OUT_PATH = "hdf5/r{0:04d}/XFEL-r{0:04d}-c{1:02d}.h5"
DATA_PATH = "entry_1/instrument_1/detector_1/detector_corrected/data"
TRAIN_PATH = "/instrument/trainID"
PULSE_PATH = "/instrument/pulseID"
CORES_COUNT = cpu_count()
WORKER_WRITE_PATH = os.path.join(os.path.dirname(__file__), '../mpi_worker_write.py')
WORKER_READ_PATH = os.path.join(os.path.dirname(__file__), '../mpi_worker_read.py')
BG_ROI = (slice(5000), slice(None))
PUPIL_ROI = (slice(750, 1040), slice(780, 1090))

GAINS = {68.8, 1.376}
GAIN_VERGE = 6000
AGIPD_GEOM = load_crystfel_geometry(os.path.join(os.path.dirname(__file__), "agipd.geom"))

def add_data_to_dset(dset, data):
    dset.refresh()
    dsetshape = dset.shape
    dset.resize((dsetshape[0] + data.shape[0],) + dsetshape[1:])
    dset[dsetshape[0]:] = data
    dset.flush()

def chunkify(start, end, thread_num=CORES_COUNT):
    limits = np.linspace(start, end, thread_num + 1).astype(int)
    return list(zip(limits[:-1], limits[1:]))

def chunkify_mpi(data_size, n_procs):
    limits = np.linspace(0, data_size, n_procs + 1).astype(int)
    return list(zip(limits[:-1], limits[1:]))

def apply_agipd_geom(frame):
    return apply_geometry_to_data(frame, AGIPD_GEOM)

def make_output_dir(path):
    try:
        os.makedirs(path)
    except OSError as error:
        if error.errno != errno.EEXIST:
            raise OSError(error.errno, error.strerror, error.filename)

def data_mpi(cheetah_path, data_size, n_procs, lim=20000):
    ranges = chunkify_mpi(data_size, n_procs - 1)
    pool = MPIPool(WORKER_READ_PATH, [cheetah_path, str(lim)], n_procs)
    return pool.read_map(ranges)

def write_args(cheetah_path, output_path, lim):
    make_output_dir(output_path)
    outfile = h5py.File(output_path, 'r+')
    arggroup = outfile.create_group('arguments')
    arggroup.create_dataset('cheetah path', data=np.string_(cheetah_path))
    arggroup.create_dataset('trimming limit', data=lim)
    outfile.close()

def write_mpi(cheetah_path, output_path, data_size, n_procs, lim=20000):
    ranges = chunkify_mpi(data_size, n_procs - 1)
    pool = MPIPool(WORKER_WRITE_PATH, [cheetah_path, output_path, str(lim)], n_procs)
    pool.write_map(ranges)
    write_args(cheetah_path, output_path, lim)

class MPIPool(object):
    def __init__(self, workerpath, args, n_procs):
        self.n_procs, self.n_workers = n_procs, n_procs - 1
        self.time = MPI.Wtime()
        self.comm = MPI.COMM_SELF.Spawn(sys.executable, args=[workerpath] + args, maxprocs=self.n_workers)

    def shutdown(self):
        self.comm.Disconnect()
        print('Elapsed time: {:.2f}s'.format(MPI.Wtime() - self.time))

    def read_map(self, task_list):
        status, queue = MPI.Status(), []
        for task in task_list[:self.n_workers]:
            self.comm.recv(source=MPI.ANY_SOURCE, status=status, tag=0)
            self.comm.send(obj=task, dest=status.Get_source())
            queue.append(status.Get_source())
        pool_size = sum([self.comm.recv(source=rank, tag=1) for rank in queue])
        for counter in range(pool_size):
            percent = (counter * 100) // pool_size
            print('\rProgress: [{0:<50}] {1:3d}%'.format('=' * (percent // 2), percent), end='\0')
            self.comm.recv(source=MPI.ANY_SOURCE, tag=2)
        print('\rProgress: [{0:<50}] {1:3d}%'.format('=' * 50, 100))
        data_list, tids_list, pids_list = [], [], []
        for rank in queue:
            data, tids, pids = self.comm.recv(source=rank, tag=3)
            data_list.append(data); tids_list.append(tids); pids_list.append(pids)
        self.shutdown()
        return data_list, tids_list, pids_list

    def write_map(self, task_list):
        status, queue = MPI.Status(), []
        for task in task_list[:self.n_workers]:
            self.comm.recv(source=MPI.ANY_SOURCE, status=status, tag=0)
            self.comm.send(obj=task, dest=status.Get_source())
            queue.append(status.Get_source())
        pool_size = sum([self.comm.recv(source=rank, tag=1) for rank in queue])
        for counter in range(pool_size):
            percent = (counter * 100) // pool_size
            print('\rProgress: [{0:<50}] {1:3d}%'.format('=' * (percent // 2), percent), end='\0')
            sys.stdout.flush()
            self.comm.recv(source=MPI.ANY_SOURCE, tag=2)
        print('\rProgress: [{0:<50}] {1:3d}%'.format('=' * 50, 100))
        sys.stdout.flush()
        data_size = 0
        for rank in queue:
            chunk_size = self.comm.sendrecv(data_size, dest=rank, source=rank)
            data_size += chunk_size
        self.comm.bcast(obj=data_size, root=MPI.ROOT)
        self.comm.Barrier()
        print('Writing data...')
        for counter in range(pool_size):
            percent = (counter * 100) // pool_size
            print('\rProgress: [{0:<50}] {1:3d}%'.format('=' * (percent // 2), percent), end='\0')
            sys.stdout.flush()
            self.comm.recv(source=MPI.ANY_SOURCE, tag=3)
        print('\rProgress: [{0:<50}] {1:3d}%'.format('=' * 50, 100))
        sys.stdout.flush()
        time.sleep(0.1)
        self.shutdown()
