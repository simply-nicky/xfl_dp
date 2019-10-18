import os
import h5py
import numpy as np
import concurrent.futures
import matplotlib.pyplot as plt
import argparse

RAW_PATH = '/gpfs/exfel/exp/MID/201901/p002543/raw/r{run_number:04d}'
EPIX_FILENAME = 'RAW-R{run_number:04d}-EPIX{epix_id:02d}'
DATA_PATH = '/INSTRUMENT/MID_EXP_EPIX-{epix_id:d}/DET/RECEIVER:daqOutput/data/image/pixels'
EPIX_DARK = '/gpfs/exfel/exp/MID/201901/p002543/usr/Shared/ePix{epix_id:02d}-r0035.h5'
OFFSET_PATH = 'darks'
MASK_PATH = 'mask'

def get_raw_data(file_path, epix_id):
    with h5py.File(file_path, 'r') as raw_file:
        raw_data = raw_file[DATA_PATH.format(epix_id=epix_id)][:]
    return raw_data

def hist(run_number, epix_id, roi=(0, 30)):
    print('Opening dark calibration file: {}'.format(EPIX_DARK.format(epix_id=epix_id - 1)))
    dark = h5py.File(EPIX_DARK.format(epix_id=epix_id - 1), 'r')
    offset = dark['darks'][:]
    mask = dark['mask'][:]
    raw_base = RAW_PATH.format(run_number=run_number)
    print('Openning EPIX files at the path: {}'.format(raw_base))
    raw_files = [filename
                 for filename in os.listdir(raw_base)
                 if filename.startswith(EPIX_FILENAME.format(run_number=run_number,
                                                             epix_id=epix_id))]
    print('EPIX files: {}'.format(raw_files))
    futures = []
    print('Reading data')
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for raw_file in raw_files:
            raw_path = os.path.join(raw_base, raw_file)
            futures.append(executor.submit(get_raw_data, raw_path, epix_id))
    raw_data = np.concatenate([future.result() for future in futures], axis=0)
    print('Raw data shape: {}'.format(raw_data.shape))
    print('Applying dark calibration data')
    data = (raw_data - offset) * mask
    print('Making histogram in {} keV energy range'.format(roi))
    hist_vals, energies = np.histogram(data.ravel(), 100, range=roi)
    energies = (energies[1:] + energies[:-1]) / 2
    hist_vals[hist_vals == 0] = 1
    fig, ax = plt.subplots(1, 1, figsize=(9, 16))
    ax.plot(energies, np.log(hist_vals))
    ax.set_xlabel('Energy, [keV]')
    fig.suptitle('Histogram')
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Plotting EPIX data histograms')
    parser.add_argument('run_number', type=int, help='run number')
    parser.add_argument('--epix_id', type=int, default=2, help='EPIX detector id number')
    args = parser.parse_args()

    hist(args.run_number, args.epix_id)

if __name__ == "__main__":
    main()
