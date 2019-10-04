"""
mpi_worker_write.py - MPI worker module for writing data
"""
import sys
import h5py
from mpi4py import MPI
from .mpi_pool import data_chunk
from .utilities import chunkify

try:
    COMM = MPI.COMM.Get_parent()
    FILE_PATH = sys.argv[1]
    OUT_PATH = sys.argv[2]
    LIMIT = int(sys.argv[3])
except:
    raise ValueError('Could not connect to parent, wrong arguments')

COMM.send(obj=None, dest=0, tag=0)
start_read, stop_read = COMM.recv(source=0)
data_list, tids_list, pids_list = [], [], []
ranges = chunkify(start_read, stop_read)
COMM.send(len(ranges), dest=0, tag=1)
for start, stop in ranges:
    _data_chunk, _tids_chunk, _pids_chunk = data_chunk(start, stop, FILE_PATH, LIMIT)
    data_list.append(_data_chunk); tids_list.append(_tids_chunk); pids_list.append(_pids_chunk)
    COMM.send(obj=None, dest=0, tag=2)
start_write = COMM.recv(source=0)
COMM.send(sum([tids.size for tids in tids_list]), dest=0)
data_size = COMM.bcast(None, root=0)
data_shape = (data_size,) + data_list[0].shape[1:]
outfile = h5py.File(OUT_PATH, 'w', driver='mpio', COMM=MPI.COMM_SELF)
datagroup = outfile.create_group('data')
dataset = datagroup.create_dataset('data', shape=data_shape, dtype=data_list[0].dtype)
trainset = datagroup.create_dataset('trainID', shape=(data_size,), dtype=tids_list[0].dtype)
pulseset = datagroup.create_dataset('pulseID', shape=(data_size,), dtype=pids_list[0].dtype)
COMM.Barrier()
for data, tids, pids in zip(data_list, tids_list, pids_list):
    with dataset.collective:
        dataset[start_write:start_write + tids.size] = data
    with trainset.collective:
        trainset[start_write:start_write + tids.size] = tids
    with pulseset.collective:
        pulseset[start_write:start_write + tids.size] = pids
    start_write += tids.size
    COMM.send(obj=None, dest=0, tag=3)
outfile.close()
COMM.Disconnect()