import sys, numpy as np, h5py
from mpi4py import MPI
from src import data_chunk
from src.utils import chunkify

try:
    comm = MPI.Comm.Get_parent()
    cheetah_path = sys.argv[1]
    output_path = sys.argv[2]
    lim = int(sys.argv[3])
except:
    raise ValueError('Could not connect to parent, wrong arguments')

start_read, stop_read = comm.sendrecv(None, dest=0, source=0)
data_list, tids_list, pids_list = [], [], []
ranges = chunkify(start_read, stop_read)
comm.gather(len(ranges), root=0)
print('Ranges gathered')
comm.Barrier()
for start, stop in ranges:
    _data_chunk, _tids_chunk, _pids_chunk = data_chunk(start, stop, cheetah_path, lim)
    data_list.append(_data_chunk); tids_list.append(_tids_chunk); pids_list.append(_pids_chunk)
    comm.send(None, dest=0)
start_write = comm.recv(source=0)
comm.send(sum([tids.size for tids in tids_list]), dest=0)
data_size = comm.bcast(None, root=0)
data_shape = (data_size,) + data_list[0].shape[1:]
outfile = h5py.File(output_path, 'w', driver='mpio', comm=MPI.COMM_SELF)
datagroup = outfile.create_group('data')
dataset = datagroup.create_dataset('data', shape=data_shape, dtype=data_list[0].dtype)
trainset = datagroup.create_dataset('trainID', shape=(data_size,), dtype=tids_list[0].dtype)
pulseset = datagroup.create_dataset('pulseID', shape=(data_size,), dtype=pids_list[0].dtype)
comm.Barrier()
for data, tids, pids in zip(data_list, tids_list, pids_list):
    with dataset.collective:
        dataset[start_write:start_write + tids.size] = data
    with trainset.collective:
        trainset[start_write:start_write + tids.size] = tids
    with pulseset.collective:
        pulseset[start_write:start_write + tids.size] = pids
    start_write += tids.size
    comm.send(None, dest=0)
comm.Disconnect()
outfile.flush()
outfile.close()