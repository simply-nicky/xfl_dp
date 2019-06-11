import os, errno, numpy as np, sys, h5py
from cfelpyutils.crystfel_utils import load_crystfel_geometry
from cfelpyutils.geometry_utils import apply_geometry_to_data
from mpi4py import MPI

basepath = "/gpfs/exfel/u/scratch/MID/201802/p002200/cheetah/hdf5/r{0:04d}-{2:s}/XFEL-r{0:04d}-c{1:02d}.{3:s}"
userpath = os.path.join(os.path.dirname(__file__), "../../cheetah/XFEL-r{0:04d}-c{1:02d}.{2:s}")
outpath = "hdf5/r{0:04d}-processed/XFEL-r{0:04d}-c{1:02d}.{2:s}"
datapath = "entry_1/instrument_1/detector_1/detector_corrected/data"
trainpath = "/instrument/trainID"
pulsepath = "/instrument/pulseID"
workerwritepath = os.path.join(os.path.dirname(__file__), '../mpi_worker_write.py')
workerreadpath = os.path.join(os.path.dirname(__file__), '../mpi_worker_read.py')
bg_roi = (slice(5000), slice(None))
thread_size = 100

gains = {68.8, 1.376}
gain_verge = 6000
AGIPD_geom = load_crystfel_geometry(os.path.join(os.path.dirname(__file__), "agipd.geom"))

class worker_star(object):
    def __init__(self, worker):
        self.worker = worker
    
    def __call__(self, args):
        return self.worker(*args)

def get_data_size(cheetah_path):
    f = h5py.File(cheetah_path, 'r')
    size = f[datapath].shape[0]
    f.close()
    return size

def chunkify(start, end):
    thread_num = (end - start) // thread_size + 1
    limits = np.linspace(start, end, thread_num + 1).astype(int)
    return list(zip(limits[:-1], limits[1:]))

def splitted_chunkify(start, end, pids):
    tasklist = []
    for pid in pids:
        for a, b in chunkify(start, end):
            tasklist.append((a, b, pid))
    return tasklist

def chunkify_mpi(data_size, n_procs):
    limits = np.linspace(0, data_size, n_procs + 1).astype(int)
    return list(zip(limits[:-1], limits[1:]))

def get_path_to_data(rnum, cnum, tag, ext, online):
    return basepath.format(rnum, cnum, tag, ext) if online else userpath.format(rnum, cnum, ext)

def apply_agipd_geom(frame):
    return apply_geometry_to_data(frame, AGIPD_geom)

def make_output_dir(path):
    try:
        os.makedirs(os.path.dirname(path))
    except OSError as e:
        if e.errno != errno.EEXIST: raise

def output_path(rnum, cnum, ext, output_folder=os.path.dirname(os.path.dirname(__file__))):
    abspath = os.path.join(output_folder, outpath.format(rnum, cnum, ext))
    if not os.path.isfile(abspath):
        return abspath
    else:
        return output_path(rnum, cnum + 1, ext, output_folder)

class MPIPool(object):
    def __init__(self, workerpath, args, n_procs):
        self.n_procs, self.n_workers = n_procs, n_procs - 1
        self.time = MPI.Wtime()
        self.comm = MPI.COMM_SELF.Spawn(sys.executable, args=[workerpath] + args, maxprocs=self.n_workers)

    def shutdown(self, queue):
        self.comm.Disconnect()
        print('Elapsed time: {:.2f}s'.format(MPI.Wtime() - self.time))

    def read_map(self, task_list):
        status, queue = MPI.Status(), []
        for task in task_list[:self.n_workers]:
            self.comm.recv(source=MPI.ANY_SOURCE, status=status, tag=0)
            self.comm.send(obj=task, dest=status.Get_source())
            queue.append(status.Get_source())
        pool_size = sum([self.comm.recv(source=rank, tag=1) for rank in queue])
        for counter in range(pool_size):
            percent = (counter * 100) // pool_size
            print('\rProgress: [{0:<50}] {1:3d}%'.format('=' * (percent // 2), percent), end='\0')
            self.comm.recv(source=MPI.ANY_SOURCE, tag=2)
        else:
            print('\rProgress: [{0:<50}] {1:3d}%'.format('=' * 50, 100))
        data_list, tids_list, pids_list = [], [], []
        for rank in queue:
            data, tids, pids = self.comm.recv(source=rank, tag=3)
            data_list.append(data); tids_list.append(tids); pids_list.append(pids)
        self.shutdown(queue)
        return data_list, tids_list, pids_list

    def write_map(self, task_list):
        status, queue = MPI.Status(), []
        for task in task_list[:self.n_workers]:
            self.comm.recv(source=MPI.ANY_SOURCE, status=status, tag=0)
            self.comm.send(obj=task, dest=status.Get_source())
            queue.append(status.Get_source())
        pool_size = sum([self.comm.recv(source=rank, tag=1) for rank in queue])
        for counter in range(pool_size):
            percent = (counter * 100) // pool_size
            print('\rProgress: [{0:<50}] {1:3d}%'.format('=' * (percent // 2), percent), end='\0')
            sys.stdout.flush()
            self.comm.recv(source=MPI.ANY_SOURCE, tag=2)
        else:
            print('\rProgress: [{0:<50}] {1:3d}%'.format('=' * 50, 100))
            sys.stdout.flush()
        data_size = 0
        for rank in queue:
            chunk_size = self.comm.sendrecv(data_size, dest=rank, source=rank)
            data_size += chunk_size
        self.comm.bcast(obj=data_size, root=MPI.ROOT)
        self.comm.Barrier()
        print('Writing data...')
        for counter in range(pool_size):
            percent = (counter * 100) // pool_size
            print('\rProgress: [{0:<50}] {1:3d}%'.format('=' * (percent // 2), percent), end='\0')
            sys.stdout.flush()
            self.comm.recv(source=MPI.ANY_SOURCE, tag=3)
        else:
            print('\rProgress: [{0:<50}] {1:3d}%'.format('=' * 50, 100))
            sys.stdout.flush()
        self.shutdown(queue)