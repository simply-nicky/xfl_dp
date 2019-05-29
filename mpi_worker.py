import sys, numpy as np
from mpi4py import MPI
from src import data_chunk

try:
    cheetah_path = sys.argv[1]
    lim = int(sys.argv[2])
    comm = MPI.Comm.Get_parent()
except:
    raise ValueError('Could not connect to parent, wrong arguments')

_data = []
for task in iter(lambda: comm.sendrecv(None, dest=0, source=0), StopIteration):
    _start, _stop = task
    _data_chunk, _pids_chunk, _tids_chunk = data_chunk(_start, _stop, cheetah_path, lim)
    _data.extend(_data_chunk)
    # _data.extend([np.arange(num) for num in range(_start, _stop)])

_data = np.array(_data)
# print(_data.shape)
comm.gather(_data, root=0)
comm.Disconnect()