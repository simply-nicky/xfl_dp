import numpy as np, h5py, concurrent.futures
from multiprocessing import cpu_count
from functools import partial
from . import utils

def get_data_shape(rnum, cnum, ext='cxi', online=True):
    path = utils.basepath if online else utils.userpath
    f = h5py.File(path.format(rnum, cnum, ext), 'r')
    shape = f[utils.datapath].shape
    f.close()
    return shape

def read_dset_shape(dset):
    dset.refresh()
    return dset.shape

def add_data_to_dset(dset, data, dsetshape):
    dset.resize((dsetshape[0] + data.shape[0],) + dsetshape[1:])
    dset[dsetshape[0]:] = data
    dset.flush()

def get_first_image(rnum, cnum, ext, bg_roi, lim, online):
    path = utils.basepath if online else utils.userpath
    file_handler = h5py.File(path.format(rnum, cnum, ext), 'r')
    pulse_ids = file_handler[utils.pulsepath]
    train_ids = file_handler[utils.trainpath]
    raw_data = file_handler[utils.datapath]
    idx = 0
    while raw_data[idx].max() < lim: idx += 1
    else:
        pid, tid = pulse_ids[idx], train_ids[idx]
        bg = raw_data[idx][bg_roi].sum().astype(np.float32)
        data = raw_data[idx].astype(np.float32) / bg
    file_handler.close()
    return data[np.newaxis], np.array([tid], dtype=np.uint32), np.array([pid], dtype=np.uint32), idx + 1

def data_chunk(start, stop, rnum, cnum, ext, bg_roi, lim, online):
    path = utils.basepath if online else utils.userpath
    file_handler = h5py.File(path.format(rnum, cnum, ext), 'r')
    pulse_ids = file_handler[utils.pulsepath]
    train_ids = file_handler[utils.trainpath]
    raw_data = file_handler[utils.datapath]
    data, tidslist, pidslist = [], [], []
    for idx in range(start, stop):
        frame = raw_data[idx]
        if frame.max() > lim:
            pidslist.append(pulse_ids[idx])
            tidslist.append(train_ids[idx])
            bg = frame[bg_roi].sum().astype(np.float32)
            data.append(frame / bg)
    return np.array(data, dtype=np.float32), np.array(tidslist, dtype=np.uint32), np.array(pidslist, dtype=np.uint32)

def data(rnum, cnum, ext='cxi', bg_roi=(slice(5000), slice(None)), lim=500, online=True):
    shape = get_data_shape(rnum, cnum, ext, online)
    worker = partial(data_chunk, rnum=rnum, cnum=cnum, ext=ext, bg_roi=bg_roi, lim=lim, online=online)
    nums = np.linspace(0, shape[0], cpu_count() + 1).astype(int)
    datalist, tidslist, pidslist = [], [], []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for data, pids, tids in executor.map(utils.worker_star(worker), zip(nums[:-1], nums[1:])):
            datalist.extend(data)
            tidslist.extend(tids)
            pidslist.extend(pids)
    return np.array(datalist, dtype=np.float32), np.array(tidslist, dtype=np.uint32), np.array(pidslist, dtype=np.uint32)

def data_serial(rnum, cnum, ext='cxi', bg_roi=(slice(5000), slice(None)), lim=500, online=True):
    path = utils.basepath if online else utils.userpath
    file_handler = h5py.File(path.format(rnum, cnum, ext), 'r', driver='family')
    pulse_ids = file_handler[utils.pulsepath]
    train_ids = file_handler[utils.trainpath]
    raw_data = file_handler[utils.datapath]
    data, tidslist, pidslist = [], [], []
    for frame, tidx, pidx in zip(raw_data, train_ids, pulse_ids):
        if frame.max() > lim:
            pidslist.append(pidx)
            tidslist.append(tidx)
            bg = frame[bg_roi].sum().astype(np.float32)
            data.append(frame / bg)
    file_handler.close()
    return np.array(data, dtype=np.float32), np.array(tidslist, dtype=np.uint32), np.array(pidslist, dtype=np.uint32)

def write_data(rnum, cnum, ext='cxi', bg_roi=(slice(5000), slice(None)), lim=500, online=True):
    shape = get_data_shape(rnum, cnum, ext, online)
    frame, tid, pid, idx = get_first_image(rnum, cnum, ext, bg_roi, lim, online)
    outfile = h5py.File(utils.outpath.format(rnum, cnum, ext), 'w', libver='latest')
    datagroup = outfile.create_group('data')
    dataset = datagroup.create_dataset('data', chunks=(1,) + shape[1:], maxshape=(None,) + shape[1:], data=frame, dtype=np.float32)
    trainset = datagroup.create_dataset('trainID', chunks=True, maxshape=(None,), data=tid, dtype=np.uint32)
    pulseset = datagroup.create_dataset('pulseID', chunks=True, maxshape=(None,), data=pid, dtype=np.uint32)
    outfile.swmr_mode = True
    worker = partial(data_chunk, rnum=rnum, cnum=cnum, ext=ext, bg_roi=bg_roi, lim=lim, online=online)
    nums = np.linspace(idx, shape[0], cpu_count() + 1).astype(int)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for data, tids, pids in executor.map(utils.worker_star(worker), zip(nums[:-1], nums[1:])):
            dshape, tshape, pshape = map(read_dset_shape, (dataset, trainset, pulseset))
            add_data_to_dset(dataset, data, dshape)
            add_data_to_dset(trainset, tids, tshape)
            add_data_to_dset(pulseset, pids, pshape)
    outfile.close()