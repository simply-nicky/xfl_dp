import numpy as np, h5py, os
from scipy.optimize import curve_fit

def gauss(xs, A, mu, sigma):
    return A * np.exp(-(xs - mu)**2 / 2 / sigma**2)

def r_to_slice(roi, roi0):
    return slice(roi[0] - roi0[0], roi[1] - roi0[0])

def onephotonpeak(data, fullroi=(-50, 200), roizero=(-15, 30), roione=(50, 100), initpar=[10, 15]):
    """
    Find one photon peak position in histogram based on fitting Gaussians.

    data - AGIPD from one memory cell (one Pulse ID)
    fullroi - histogram ROI of ADU values
    roizero - ROI of zero photon peak
    roione - ROI of one photon peak
    initpar - initial parameters of gaussian for fitting
    """
    # making histogram
    hist, bin_edges = np.histogram(data.ravel(), fullroi[1] - fullroi[0], range=fullroi)
    # Supressing 0 ADU value peak
    hist[abs(fullroi[0])] = (hist[abs(fullroi[0]) - 1] + hist[abs(fullroi[0]) + 1]) / 2
    # Making log histogram
    hlog, xs = np.log(hist) - np.log(hist).min(), (bin_edges[1:] + bin_edges[:-1]) / 2
    # Finding zero photon peak
    mu1 = xs[hlog[r_to_slice(roizero, fullroi)].argmax() + roizero[0] - fullroi[0]]
    # Fitting gaussian ro one photon peak
    p1, _ = curve_fit(lambda x, A, sigma: gauss(x, A, mu1, sigma), xs[r_to_slice(roione, fullroi)], hlog[r_to_slice(roione, fullroi)], p0=initpar)
    # Finding one photon peak
    mu2 = xs[(hlog - gauss(xs, p1[0], mu1, p1[1]))[r_to_slice(roione, fullroi)].argmax() + roione[0] - fullroi[0]]
    return mu2 - mu1

def relativegain(ADUs, pid, dig_gains, gainlevels, offsets, badpixels, roi=[0, 60]):
    """
    Return relative gain value based on histogram of High gain ADUs divided by Medium gain ADUs.

    ADUs - AGIPD raw adu values of one memory cell (Pulse ID = pid)
    pid - Pulse ID
    dig_gains - AGIPD digital gain values
    gainlevels - calibration gain levels
    offsets - callibration ADU offsets
    badpixels - bad pixels mask
    """
    # memory cell id
    mid = pid // 4
    # gain mode mask
    gainmode = np.array([frame > gainlevels[1, mid] for frame in dig_gains], dtype=int)
    # high gain ADUs
    highgains = np.where((gainmode == 0) & (badpixels[0, mid] == 0), ADUs - offsets[0, mid], 0)
    highgains[highgains < 0] = 0
    # medium gain ADUs
    midgains = np.where((gainmode == 1) & (badpixels[1, mid] == 0), ADUs - offsets[1, mid], 0)
    midgains[midgains < 0] = 0
    # Finding mean ADU value for high gain ADUs
    htotals = np.sum(np.array(highgains != 0, dtype=int), axis=0)
    avhgains = np.where(htotals != 0, highgains.sum(axis=0) / htotals, 0)
    # Finding mean ADU value for high gain ADUs
    mtotals = np.sum(np.array(midgains != 0, dtype=int), axis=0)
    avmgains = np.where(mtotals != 0, midgains.sum(axis=0) / mtotals, 0)
    # Division of mean high gain ADUs and medium gain ADUs gives us relative gain value
    relgains = np.where((avhgains != 0) & (avmgains != 0), avhgains / avmgains, 0)
    # Finding relative gain value by fitting gaussian to histgram
    hist, bin_edges = np.histogram(relgains.ravel(), 100, range=(0.5, 100.5))
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
    p, _ = curve_fit(lambda x, mu, sigma: gauss(x, hist.max(), mu, sigma), bin_centres[roi[0]:roi[1]], hist[roi[0]:roi[1]], bounds=([roi[0], 0], [roi[1], 10]))
    return p[0]

if __name__ == "__main__":
    from timeit import default_timer as timer
    outpath = "hdf5/r{0:04d}-processed/XFEL-r{0:04d}-c{1:02d}.cxi"
    datapath = 'data/data'
    
    f283 = h5py.File(outpath.format(283, 0), 'r')
    datalist_283 = [f283[os.path.join(datapath, key)][:] for key in f283[datapath]]
    time = timer()
    gain = np.array([onephotonpeak(data) for data in datalist_283]).mean()
    print('{:.2f}s'.format(timer() - time))

    f221raw = h5py.File('raw/r0221_raw.h5', 'r')
    adus = f221raw['data/ADU'][:]
    gains = f221raw['data/DigitalGains'][:]

    # Opening cheetah calibration files
    calib = h5py.File('cheetah/calib/Cheetah-calib-pupil.h5', 'r')
    gainlevels = calib['DigitalGainLevel']
    offsets = calib['AnalogOffset']
    badpixels = calib['Badpixel']
    
    time = timer()
    relgain = relativegain(adus, 4, gains, gainlevels, offsets, badpixels)
    print('{:.2f}s'.format(timer() - time))

    print(gain, relgain)