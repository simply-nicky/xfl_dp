import sys, numpy as np, h5py
from mpi4py import MPI
from src import data_chunk
from src.utils import chunkify

try:
    comm = MPI.Comm.Get_parent()
    cheetah_path = sys.argv[1]
    lim = int(sys.argv[2])
except:
    raise ValueError('Could not connect to parent, wrong arguments')

comm.send(obj=None, dest=0, tag=0)
start_read, stop_read = comm.recv(source=0)
print('{:d} received'.format(comm.Get_rank()))
data_list, tids_list, pids_list = [], [], []
ranges = chunkify(start_read, stop_read)
comm.send(len(ranges), dest=0, tag=1)
print('{:d} gathered'.format(comm.Get_rank()))
for start, stop in ranges:
    _data_chunk, _tids_chunk, _pids_chunk = data_chunk(start, stop, cheetah_path, lim)
    data_list.append(_data_chunk); tids_list.append(_tids_chunk); pids_list.append(_pids_chunk)
    print('{:d} data processed'.format(comm.Get_rank()))
    comm.isend(obj=None, dest=0)
    print('{:d} sent in loop'.format(comm.Get_rank()))
data = np.concatenate(data_list); tids = np.concatenate(tids_list); pids = np.concatenate(pids_list)
req = comm.isend(obj=(data, tids, pids), dest=0, tag=3)
req.wait()
print('finished')
comm.Disconnect()