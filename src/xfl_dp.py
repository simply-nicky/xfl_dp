import numpy as np, h5py, argparse, concurrent.futures, sys
from mpi4py import MPI
from functools import partial
from . import utils

def get_data_size(cheetah_path):
    f = h5py.File(cheetah_path, 'r')
    size = f[utils.datapath].shape[0]
    f.close()
    return size

def add_data_to_dset(dset, data):
    dset.refresh()
    dsetshape = dset.shape
    dset.resize((dsetshape[0] + data.shape[0],) + dsetshape[1:])
    dset[dsetshape[0]:] = data
    dset.flush()

def process_frame(frame):
    return utils.apply_agipd_geom(frame).astype(np.int32)

def get_first_image(cheetah_path, lim):
    file_handler = h5py.File(cheetah_path, 'r')
    pulse_ids = file_handler[utils.pulsepath]
    train_ids = file_handler[utils.trainpath]
    raw_data = file_handler[utils.datapath]
    idx = 0
    while raw_data[idx].max() < lim: idx += 1
    else:
        pid, tid = pulse_ids[idx], train_ids[idx]
        data = process_frame(raw_data[idx])
    file_handler.close()
    return data[np.newaxis], np.array([tid]), np.array([pid]), idx + 1

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
    return data, tidslist, pidslist

def data(cheetah_path, data_size, lim=20000):
    worker = partial(data_chunk, cheetah_path=cheetah_path, lim=lim)
    nums = utils.chunkify(0, data_size)
    datalist, tidslist, pidslist = [], [], []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for data, pids, tids in executor.map(utils.worker_star(worker), zip(nums[:-1], nums[1:])):
            datalist.extend(data)
            tidslist.extend(tids)
            pidslist.extend(pids)
    return np.array(datalist), np.array(tidslist), np.array(pidslist)

def data_mpi(cheetah_path, data_size, n_procs, lim=20000):
    nums = utils.chunkify(0, data_size)
    task_list = list(zip(nums[:-1], nums[1:])) + ([StopIteration] * (n_procs - 1))
    comm = MPI.COMM_WORLD.Spawn(sys.executable, args=[utils.workerpath, cheetah_path, str(lim)], maxprocs=n_procs - 1)
    status = MPI.Status()
    for counter, task in enumerate(task_list):
        comm.recv(source=MPI.ANY_SOURCE, status=status)
        comm.send(obj=task, dest=status.Get_source())
        percent = ((counter + 1) * 100) // len(task_list)
        print('Progress: [{0:<50}] {1:3d}%'.format('=' * (percent // 2), percent))
        sys.stdout.flush()
    data = comm.gather(None, root=MPI.ROOT)
    comm.Disconnect()
    return data

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

def write_args(pids, tids, cheetah_path, output_path, lim):
    utils.make_output_dir(output_path)
    outfile = h5py.File(output_path, 'w')
    arggroup = outfile.create_group('arguments')
    arggroup.create_dataset('cheetah path', data=np.string_(cheetah_path))
    arggroup.create_dataset('trimming limit', data=lim)
    datagroup = outfile.create_group('data')
    datagroup.create_dataset('trainID', data=tids)
    datagroup.create_dataset('pulseID', data=pids)
    outfile.close()

def write_data(cheetah_path, output_path, data_size, lim=20000):
    utils.make_output_dir(output_path)
    frame, tid, pid, idx = get_first_image(cheetah_path, lim)
    outfile = h5py.File(output_path, 'w', driver='mpio', comm=MPI.COMM_WORLD)
    arggroup = outfile.create_group('arguments')
    arggroup.create_dataset('cheetah path', data=np.string_(cheetah_path))
    arggroup.create_dataset('trimming limit', data=lim)
    datagroup = outfile.create_group('data')
    dataset = datagroup.create_dataset('data', chunks=frame.shape, maxshape=(None,) + frame.shape[1:], data=frame, compression='gzip')
    trainset = datagroup.create_dataset('trainID', chunks=True, maxshape=(None,), data=tid)
    pulseset = datagroup.create_dataset('pulseID', chunks=True, maxshape=(None,), data=pid)
    outfile.swmr_mode = True
    worker = partial(data_chunk, cheetah_path=cheetah_path, lim=lim)
    nums = utils.chunkify(idx, data_size)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for data, tids, pids in executor.map(utils.worker_star(worker), zip(nums[:-1], nums[1:])):
            add_data_to_dset(dataset, data)
            add_data_to_dset(trainset, tids)
            add_data_to_dset(pulseset, pids)
    outfile.close()

def write_mpi(data, pids, tids, cheetah_path, output_path, lim=20000):
    write_args(pids, tids, cheetah_path, output_path, lim)
    rank = MPI.COMM_WORLD.rank
    size = MPI.COMM_WORLD.size
    outfile = h5py.File(output_path, 'r+', driver='mpio', comm=MPI.COMM_WORLD)
    dataset = outfile.create_dataset('data/data', data.shape, dtype=data.dtype)
    start = int(data.shape[0] / size * rank)
    stop = int(data.shape[0] / size * (rank + 1))
    dataset[start:stop] = data[start:stop]
    outfile.close()

def splitted_data_chunk(start, stop, cheetah_path, pids, lim):
    file_handler = h5py.File(cheetah_path, 'r')
    pulse_ids = file_handler[utils.pulsepath]
    train_ids = file_handler[utils.trainpath]
    raw_data = file_handler[utils.datapath]
    datalist, tidslist, pidslist = [[] for _ in range(len(pids))], [[] for _ in range(len(pids))], list(pids)
    for idx in range(start, stop):
        if raw_data[idx].max() > lim and pulse_ids[idx] in pids:
            new_idx = pidslist.index(pulse_ids[idx])
            datalist[new_idx].append(process_frame(raw_data[idx]))
            tidslist[new_idx].append(train_ids[idx])
    return datalist, tidslist

def splitted_data(cheetah_path, data_size, pids, lim=20000):
    worker = partial(splitted_data_chunk, cheetah_path=cheetah_path, lim=lim, pids=pids)
    nums = utils.chunkify(0, data_size)
    datalist, tidslist = [[] for _ in range(len(pids))], [[] for _ in range(len(pids))]
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for datalist_chunk, tidslist_chunk in executor.map(utils.worker_star(worker), zip(nums[:-1], nums[1:])):
            for (idx, data), tids in zip(enumerate(datalist_chunk), tidslist_chunk):
                datalist[idx].extend(data)
                tidslist[idx].extend(tids)
    return [np.array(data) for data in datalist], [np.array(tids) for tids in tidslist]

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