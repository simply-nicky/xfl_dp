"""
mpi_pool.py - MPI pool implementation module
"""
import time
import sys
import os
import h5py
import numpy as np
from mpi4py import MPI
from .utilities import apply_agipd_geom, make_output_dir

DATA_PATH = "entry_1/instrument_1/detector_1/detector_corrected/data"
TRAIN_PATH = "/instrument/trainID"
PULSE_PATH = "/instrument/pulseID"
WORKER_WRITE_PATH = os.path.join(os.path.dirname(__file__), '../mpi_worker_write.py')
WORKER_READ_PATH = os.path.join(os.path.dirname(__file__), '../mpi_worker_read.py')

def chunkify_mpi(data_size, n_procs):
    limits = np.linspace(0, data_size, n_procs + 1).astype(int)
    return list(zip(limits[:-1], limits[1:]))

def process_frame(frame):
    return apply_agipd_geom(frame).astype(np.int32)

def data_chunk(start, stop, cheetah_path, lim):
    file_handler = h5py.File(cheetah_path, 'r')
    pulse_ids = file_handler[PULSE_PATH]
    train_ids = file_handler[TRAIN_PATH]
    raw_data = file_handler[DATA_PATH]
    data, tidslist, pidslist = [], [], []
    for idx in range(start, stop):
        if raw_data[idx].max() > lim:
            pidslist.append(pulse_ids[idx])
            tidslist.append(train_ids[idx])
            data.append(process_frame(raw_data[idx]))
    return np.array(data), np.array(tidslist), np.array(pidslist)

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
        self.comm = MPI.COMM_SELF.Spawn(sys.executable,
                                        args=[workerpath] + args,
                                        maxprocs=self.n_workers)

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
            data_list.append(data)
            tids_list.append(tids)
            pids_list.append(pids)
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
