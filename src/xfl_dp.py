import numpy as np, h5py, argparse, concurrent.futures, sys
from mpi4py import MPI
from functools import partial
from . import utils

def process_frame(frame):
    return utils.apply_agipd_geom(frame).astype(np.int32)

def data_chunk(start, stop, cheetah_path, lim):
    file_handler = h5py.File(cheetah_path, 'r')
    pulse_ids = file_handler[utils.pulsepath]
    train_ids = file_handler[utils.trainpath]
    raw_data = file_handler[utils.datapath]
    data, tidslist, pidslist = [], [], []
    for idx in range(start, stop):
        if raw_data[idx].max() > lim:
            pidslist.append(pulse_ids[idx])
            tidslist.append(train_ids[idx])
            data.append(process_frame(raw_data[idx]))
    return np.array(data), np.array(tidslist), np.array(pidslist)

def data_mpi(cheetah_path, data_size, n_procs, lim=20000):
    ranges = utils.chunkify_mpi(data_size, n_procs - 1)
    pool = utils.MPIPool(utils.workerreadpath, [cheetah_path, str(lim)], n_procs)
    return pool.read_map(ranges)

def write_args(cheetah_path, output_path, lim):
    utils.make_dirs(output_path)
    outfile = h5py.File(output_path, 'r+')
    arggroup = outfile.create_group('arguments')
    arggroup.create_dataset('cheetah path', data=np.string_(cheetah_path))
    arggroup.create_dataset('trimming limit', data=lim)
    outfile.close()

def write_mpi(cheetah_path, output_path, data_size, n_procs, lim=20000):
    ranges = utils.chunkify_mpi(data_size, n_procs - 1)
    pool = utils.MPIPool(utils.workerwritepath, [cheetah_path, output_path, str(lim)], n_procs)
    pool.write_map(ranges)
    write_args(cheetah_path, output_path, lim)

################################################################
# ---------------------- Serial methods ---------------------- #
################################################################

def splitted_data_serial(cheetah_path, pids, lim=20000):
    file_handler = h5py.File(cheetah_path, 'r')
    pulse_ids = file_handler[utils.pulsepath]
    train_ids = file_handler[utils.trainpath]
    raw_data = file_handler[utils.datapath]
    datalist, tidslist, pidslist = [[] for _ in range(len(pids))], [[] for _ in range(len(pids))], list(pids)
    for frame, tidx, pidx in zip(raw_data, train_ids, pulse_ids):
        if frame.max() > lim and pidx in pids:
            new_idx = pidslist.index(pidx)
            datalist[new_idx].append(process_frame(frame))
            tidslist[new_idx].append(tidx)
    file_handler.close()
    return [np.array(data) for data in datalist], [np.array(tids) for tids in tidslist]

def data_serial(cheetah_path, lim=20000):
    file_handler = h5py.File(cheetah_path, 'r')
    pulse_ids = file_handler[utils.pulsepath]
    train_ids = file_handler[utils.trainpath]
    raw_data = file_handler[utils.datapath]
    data, tidslist, pidslist = [], [], []
    for frame, tidx, pidx in zip(raw_data, train_ids, pulse_ids):
        if frame.max() > lim:
            pidslist.append(pidx)
            tidslist.append(tidx)
            data.append(process_frame(frame))
    file_handler.close()
    return np.array(data), np.array(tidslist), np.array(pidslist)