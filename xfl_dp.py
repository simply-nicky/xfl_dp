import numpy as np, h5py, concurrent.futures, argparse
from multiprocessing import cpu_count
from functools import partial
from . import utils

def get_data_size(rnum, cnum, tag, ext, online):
    f = h5py.File(utils.get_path_to_data(rnum, cnum, tag, ext, online), 'r')
    size = f[utils.datapath].shape[0]
    f.close()
    return size

def add_data_to_dset(dset, data):
    dset.refresh()
    dsetshape = dset.shape
    dset.resize((dsetshape[0] + data.shape[0],) + dsetshape[1:])
    dset[dsetshape[0]:] = data
    dset.flush()

def get_first_image(rnum, cnum, tag, ext, bg_roi, lim, online):
    file_handler = h5py.File(utils.get_path_to_data(rnum, cnum, tag, ext, online), 'r')
    pulse_ids = file_handler[utils.pulsepath]
    train_ids = file_handler[utils.trainpath]
    raw_data = file_handler[utils.datapath]
    size = raw_data.shape[0]
    idx = 0
    while raw_data[idx].max() < lim: idx += 1
    else:
        pid, tid = pulse_ids[idx], train_ids[idx]
        bg = raw_data[idx][bg_roi].sum().astype(np.float32)
        data = utils.apply_agipd_geom(raw_data[idx]).astype(np.float32) / bg
    file_handler.close()
    return data[np.newaxis], np.array([tid], dtype=np.uint32), np.array([pid], dtype=np.uint32), size, idx + 1

def data_chunk(start, stop, rnum, cnum, tag, ext, bg_roi, lim, online):
    file_handler = h5py.File(utils.get_path_to_data(rnum, cnum, tag, ext, online), 'r')
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
            data.append(utils.apply_agipd_geom(frame).astype(np.float32) / bg)
    return np.array(data, dtype=np.float32), np.array(tidslist, dtype=np.uint32), np.array(pidslist, dtype=np.uint32)

def data(rnum, cnum, tag, ext='cxi', bg_roi=(slice(5000), slice(None)), lim=500, online=True):
    size = get_data_size(rnum, cnum, tag, ext, online)
    worker = partial(data_chunk, rnum=rnum, cnum=cnum, ext=ext, bg_roi=bg_roi, lim=lim, online=online)
    nums = np.linspace(0, size, cpu_count() + 1).astype(int)
    datalist, tidslist, pidslist = [], [], []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for data, pids, tids in executor.map(utils.worker_star(worker), zip(nums[:-1], nums[1:])):
            datalist.extend(data)
            tidslist.extend(tids)
            pidslist.extend(pids)
    return np.array(datalist, dtype=np.float32), np.array(tidslist, dtype=np.uint32), np.array(pidslist, dtype=np.uint32)

def data_serial(rnum, cnum, tag, ext='cxi', bg_roi=(slice(5000), slice(None)), lim=500, online=True):
    file_handler = h5py.File(utils.get_path_to_data(rnum, cnum, tag, ext, online), 'r')
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

def write_data(rnum, cnum, tag, ext='cxi', bg_roi=(slice(5000), slice(None)), lim=500, online=True):
    path = utils.output_path(rnum, cnum, ext)
    utils.make_output_dir(path)
    frame, tid, pid, size, idx = get_first_image(rnum, cnum, tag, ext, bg_roi, lim, online)
    outfile = h5py.File(path, 'w', libver='latest')
    datagroup = outfile.create_group('data')
    dataset = datagroup.create_dataset('data', chunks=frame.shape, maxshape=(None,) + frame.shape[1:], data=frame, dtype=np.float32)
    trainset = datagroup.create_dataset('trainID', chunks=True, maxshape=(None,), data=tid, dtype=np.uint32)
    pulseset = datagroup.create_dataset('pulseID', chunks=True, maxshape=(None,), data=pid, dtype=np.uint32)
    outfile.swmr_mode = True
    worker = partial(data_chunk, rnum=rnum, cnum=cnum, tag=tag, ext=ext, bg_roi=bg_roi, lim=lim, online=online)
    nums = np.linspace(idx, size, cpu_count() + 1).astype(int)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for data, tids, pids in executor.map(utils.worker_star(worker), zip(nums[:-1], nums[1:])):
            add_data_to_dset(dataset, data)
            add_data_to_dset(trainset, tids)
            add_data_to_dset(pulseset, pids)
    outfile.close()

def main():
    parser = argparse.ArgumentParser(description='Run XFEL post processing of cheetah data')
    parser.add_argument('rnum', type=int, help='run number')
    parser.add_argument('cnum', type=int, help='cheetah number')
    parser.add_argument('tag', type=str, help='cheetah tag associated with the current run (written after hyphen in cheetah folder name)')
    parser.add_argument('limit', type=int, nargs='?', default=500, help='minimum ADU value to trim out black images')
    parser.add_argument('-v', '--verbosity', action='store_true', help='increase output verbosity')
    args = parser.parse_args()
    if args.verbosity:
        print("List of typed arguments: %s" % args)
        print("Writing data to folder: %s" % utils.get_path_to_data(args.rnum, args.cnum, args.tag, 'cxi', True))
        write_data(args.rnum, args.cnum, args.tag, 'cxi', (slice(5000), slice(None)), args.limit, True)
        print("Done")
    else:
        write_data(args.rnum, args.cnum, args.tag, 'cxi', (slice(5000), slice(None)), args.limit, True)       