"""
mpi_worker_read.py - MPI worker module for reading data
"""
import sys
import numpy as np
from mpi4py import MPI
from .mpi_pool import data_chunk
from .utilities import chunkify

try:
    COMM = MPI.COMM.Get_parent()
    FILE_PATH = sys.argv[1]
    LIMIT = int(sys.argv[2])
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
data = np.concatenate(data_list); tids = np.concatenate(tids_list); pids = np.concatenate(pids_list)
COMM.send(obj=(data, tids, pids), dest=0, tag=3)

COMM.Disconnect()