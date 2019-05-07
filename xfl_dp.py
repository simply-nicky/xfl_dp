import numpy as np, h5py, concurrent.futures, argparse
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

def process_frame(frame, normalize):
    frame[frame < 0] = 0
    if normalize:
        bg = frame[utils.bg_roi].sum().astype(np.float32)
        return utils.apply_agipd_geom(frame).astype(np.float32) / bg
    else:
        return utils.apply_agipd_geom(frame)

def get_first_image(cheetah_path, lim, normalize):
    file_handler = h5py.File(cheetah_path, 'r')
    pulse_ids = file_handler[utils.pulsepath]
    train_ids = file_handler[utils.trainpath]
    raw_data = file_handler[utils.datapath]
    idx = 0
    while raw_data[idx].max() < lim: idx += 1
    else:
        pid, tid = pulse_ids[idx], train_ids[idx]
        data = process_frame(raw_data[idx], normalize)
    file_handler.close()
    return data[np.newaxis], np.array([tid]), np.array([pid]), idx + 1

def data_chunk(start, stop, cheetah_path, lim, normalize):
    file_handler = h5py.File(cheetah_path, 'r')
    pulse_ids = file_handler[utils.pulsepath]
    train_ids = file_handler[utils.trainpath]
    raw_data = file_handler[utils.datapath]
    data, tidslist, pidslist = [], [], []
    for idx in range(start, stop):
        frame = raw_data[idx]
        if frame.max() > lim:
            pidslist.append(pulse_ids[idx])
            tidslist.append(train_ids[idx])
            data.append(process_frame(frame, normalize))
    return np.array(data), np.array(tidslist), np.array(pidslist)

def data(cheetah_path, data_size, lim=20000, normalize=True):
    worker = partial(data_chunk, cheetah_path=cheetah_path, lim=lim, normalize=normalize)
    nums = utils.chunkify(0, data_size)
    datalist, tidslist, pidslist = [], [], []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for data, pids, tids in executor.map(utils.worker_star(worker), zip(nums[:-1], nums[1:])):
            datalist.extend(data)
            tidslist.extend(tids)
            pidslist.extend(pids)
    return np.array(datalist), np.array(tidslist), np.array(pidslist)

def data_serial(cheetah_path, lim=20000, normalize=True):
    file_handler = h5py.File(cheetah_path, 'r')
    pulse_ids = file_handler[utils.pulsepath]
    train_ids = file_handler[utils.trainpath]
    raw_data = file_handler[utils.datapath]
    data, tidslist, pidslist = [], [], []
    for frame, tidx, pidx in zip(raw_data, train_ids, pulse_ids):
        if frame.max() > lim:
            pidslist.append(pidx)
            tidslist.append(tidx)
            data.append(process_frame(frame, normalize))
    file_handler.close()
    return np.array(data), np.array(tidslist), np.array(pidslist)

def write_data(cheetah_path, output_path, data_size, lim=20000, normalize=True):
    utils.make_output_dir(output_path)
    frame, tid, pid, idx = get_first_image(cheetah_path, lim, normalize)
    outfile = h5py.File(output_path, 'w', libver='latest')
    arggroup = outfile.create_group('arguments')
    arggroup.create_dataset('cheetah path', data=np.string_(cheetah_path))
    arggroup.create_dataset('trimming limit', data=lim)
    arggroup.create_dataset('normalize', data=normalize)
    datagroup = outfile.create_group('data')
    dataset = datagroup.create_dataset('data', chunks=frame.shape, maxshape=(None,) + frame.shape[1:], data=frame, dtype=np.float32)
    trainset = datagroup.create_dataset('trainID', chunks=True, maxshape=(None,), data=tid, dtype=np.uint32)
    pulseset = datagroup.create_dataset('pulseID', chunks=True, maxshape=(None,), data=pid, dtype=np.uint32)
    outfile.swmr_mode = True
    worker = partial(data_chunk, cheetah_path=cheetah_path, lim=lim, normalize=normalize)
    nums = utils.chunkify(idx, data_size)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for data, tids, pids in executor.map(utils.worker_star(worker), zip(nums[:-1], nums[1:])):
            add_data_to_dset(dataset, data)
            add_data_to_dset(trainset, tids)
            add_data_to_dset(pulseset, pids)
    outfile.close()  