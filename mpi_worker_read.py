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

start_read, stop_read = comm.sendrecv(None, dest=0, source=0)
print('{:d} received'.format(comm.Get_rank()))
data_list, tids_list, pids_list = [], [], []
ranges = chunkify(start_read, stop_read)
comm.gather(len(ranges), root=0)
print('{:d} gathered'.format(comm.Get_rank()))
for start, stop in ranges:
    _data_chunk, _tids_chunk, _pids_chunk = data_chunk(start, stop, cheetah_path, lim)
    data_list.append(_data_chunk); tids_list.append(_tids_chunk); pids_list.append(_pids_chunk)
    comm.send(None, dest=0, tag=0)
    print('{:d} sent in loop'.format(comm.Get_rank()))
data = np.concatenate(data_list); tids = np.concatenate(tids_list); pids = np.concatenate(pids_list)
req = comm.isend(obj=(data, tids, pids), dest=0, tag=1)
req.wait()
print('finished')
comm.Disconnect()