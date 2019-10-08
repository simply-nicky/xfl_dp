"""
calib.py - calibration module
"""
import numpy as np
import pyqtgraph as pg
from scipy.optimize import curve_fit

try:
    from PyQt5 import QtCore, QtGui
except ImportError:
    from PyQt4 import QtCore, QtGui

MODULE_SHAPE = (512, 128)
AGIPD_SHAPE = (8192, 128)
HG_LEVEL = 0
MG_LEVEL = 6000
LG_LEVEL = 32000
HG_GAIN = 1 / 68.8
MG_GAIN = 1 / 1.376
LG_GAIN = 5 / 1.376

def gauss(arg, amplitude, mu, sigma):
    """
    Gaussian function

    arg - argument
    amplitude - amplitude constant, max function value
    mu - center of gaussian
    sigma - gaussian width
    """
    return amplitude * np.exp(-(arg - mu)**2 / 2 / sigma**2)

class ROI(object):
    def __init__(self, lower_bound, higher_bound):
        if lower_bound > higher_bound:
            raise ValueError("Invalid bounds")
        self.lower_bound, self.higher_bound = lower_bound, higher_bound

    @property
    def length(self):
        return self.higher_bound - self.lower_bound

    @property
    def range(self):
        return (self.lower_bound, self.higher_bound)

    def relative(self, base_roi):
        return ROI(self.lower_bound - base_roi.lower_bound,
                   self.higher_bound - base_roi.lower_bound)

FULL_ROI = ROI(-50, 150)
ZERO_ROI = ROI(-50, 30).relative(FULL_ROI)
ONE_ROI = ROI(30, 150).relative(FULL_ROI)
REL_ROI = ROI(20, 60)

class CalibParameter(object):
    def __init__(self, level):
        self.module = level * np.ones(MODULE_SHAPE)
        self.agipd = level * np.ones(AGIPD_SHAPE)

class GainLevel(object):
    def __init__(self, hg_level=HG_LEVEL, mg_level=MG_LEVEL, lg_level=LG_LEVEL):
        self.hg_level = CalibParameter(hg_level)
        self.mg_level = CalibParameter(mg_level)
        self.lg_level = CalibParameter(lg_level)

    def mask_module(self, dig_gains):
        mg_mask = (dig_gains > self.mg_level.module).astype(np.uint8)
        hg_mask = (dig_gains > self.hg_level.module).astype(np.uint8)
        return mg_mask + hg_mask

    def mask_agipd(self, dig_gains):
        mg_mask = (dig_gains > self.mg_level.agipd).astype(np.uint8)
        hg_mask = (dig_gains > self.hg_level.agipd).astype(np.uint8)
        return mg_mask + hg_mask

class CalibViewer(QtGui.QMainWindow):
    def __init__(self, hist, adus, filename, parent=None, size=(1280, 720)):
        super(CalibViewer, self).__init__(parent=parent, size=QtCore.QSize(size[0], size[1]))
        self.full_roi = ROI(adus.min(), adus.max())
        self.zero_roi = ROI(self.full_roi.lower_bound,
                            self.full_roi.lower_bound + 0.4 * self.full_roi.length)
        self.one_roi = ROI(self.full_roi.higher_bound - 0.55 * self.full_roi.length,
                           self.full_roi.higher_bound - 0.05 * self.full_roi.length)
        self.init_ui(hist, adus, filename)

    def init_ui(self, hist, adus, filename):
        self.hbox_layout = QtGui.QHBoxLayout()
        label_widget = QtGui.Qlabel("ADU Histogram")
        self.hbox_layout.addWidget(label_widget)
        plot_widget = pg.PlotWidget(name="Plot", background='w')
        hist_plot = plot_widget.plot(adus, hist)
        hist_plot.setPen(color=1., width=3)
        self.zero_plot = plot_widget.plot()
        self.zero_plot.setPen(color='b', width=2, style=QtCore.DashLine)
        self.one_plot = plot_widget.plot()
        self.one_plot.setPen(color='r', width=2, style=QtCore.DashLine)
        self.hbox_layout.addWidget(plot_widget)
        vbox = QtGui.QVBoxLayout()
        update_button = QtGui.QPushButton("Update fit")
        update_button.clicked.connect(self.update_fit)
        vbox.addWidget(self.update_button)
        export_button = QtGui.QPushButton("Export")
        export_button.clicked.connect(self.export_data)
        vbox.addWidget(self.export_button)
        self.hbox_layout.addWidget(vbox)
        self.setWindowTitle('Photon Calibration, {}'.format(filename))
        self.show()

    def update_fit(self):
        pass

    def export_data(self):
        pass

    def update_plot(self):
        pass

def hg_calibrate(hg_data, full_roi=FULL_ROI, zero_roi=ZERO_ROI, one_roi=ONE_ROI):
    # making high gain histogram
    hist, hg_edges = np.histogram(hg_data.ravel(), full_roi.length, range=full_roi.range)
    # Supressing 0 ADU value peak
    zero_peak = abs(full_roi.lower_bound)
    hist[zero_peak] = (hist[zero_peak + 1] + hist[zero_peak - 1]) / 2
    # making log histogram
    log_hist, adus = np.log(hist) - np.log(hist).min(), (hg_edges[1:] + hg_edges[:-1]) / 2
    # finding zero photon peak
    zero_adu = adus[log_hist[zero_roi].argmax()]
    # fitting gaussian function to zero photon peak
    fit_pars, _ = curve_fit(lambda x, amplitude, sigma: gauss(x, amplitude, zero_adu, sigma),
                            adus[zero_roi],
                            log_hist[zero_roi])
    # finding one photon peak
    one_adu = adus[(log_hist - gauss(adus, fit_pars[0], zero_adu, fit_pars[1]))[one_roi].argmax()]
    return zero_adu, one_adu

def mg_calibrate(mg_data, hg_data, rel_roi=REL_ROI):
    hg_totals = np.sum(np.array(hg_data <= 0, dtype=np.uint32), axis=0)
    mg_totals = np.sum(np.array(mg_data <= 0, dtype=np.uint32), axis=0)
    hg_average = np.where(hg_totals != 0, hg_data.sum(axis=0) / hg_totals, 0)
    mg_average = np.where(mg_totals != 0, mg_data.sum(axis=0) / mg_totals, 0)
    rel_data = np.where(mg_average != 0, hg_average / mg_average, 0)
    rel_hist, edges = np.histogram(rel_data.ravel(), rel_roi.length, range=rel_roi.range)
    adus = (edges[:-1] + edges[1:]) / 2
    fit_par, _ = curve_fit(lambda x, mu, sigma: gauss(x, rel_hist.max(), mu, sigma),
                           adus,
                           rel_hist,
                           bounds=([rel_roi.lower_bound, 0], [rel_roi.higher_bound, 20]))
    return fit_par[0]

def calibrate_module(data, pids, gain_levels):
    data_list, dig_gain_list, _, new_pids = data.get_ordered_data(pids)
    offset_list, gain_list = [], []
    for data_chunk, dig_gain_chunk in zip(data_list, dig_gain_list):
        mask = gain_levels.mask_module(dig_gain_chunk)
        hg_data = data_chunk[mask == 0]
        zero_adu, one_adu = hg_calibrate(hg_data)
        offset_list.append(zero_adu)
        gain_list.append(one_adu - zero_adu)
    return np.array(offset_list), np.array(gain_list), new_pids

# def r_to_slice(roi, roi0):
#     return slice(roi[0] - roi0[0], roi[1] - roi0[0])

# def onephotonpeak(data, fullroi=(-50, 200), roizero=(-15, 30), roione=(50, 100), initpar=[10, 15]):
#     """
#     Find one photon peak position in histogram based on fitting Gaussians.

#     data - agipd from one memory cell (one Pulse ID)
#     fullroi - histogram ROI of ADU values
#     roizero - ROI of zero photon peak
#     roione - ROI of one photon peak
#     initpar - initial parameters of gaussian for fitting
#     """
#     # making histogram
#     hist, bin_edges = np.histogram(data.ravel(), fullroi[1] - fullroi[0], range=fullroi)
#     # Supressing 0 ADU value peak
#     hist[abs(fullroi[0])] = (hist[abs(fullroi[0]) - 1] + hist[abs(fullroi[0]) + 1]) / 2
#     # Making log histogram
#     hlog, xs = np.log(hist) - np.log(hist).min(), (bin_edges[1:] + bin_edges[:-1]) / 2
#     # Finding zero photon peak
#     mu1 = xs[hlog[r_to_slice(roizero, fullroi)].argmax() + roizero[0] - fullroi[0]]
#     # Fitting gaussian ro one photon peak
#     p1, _ = curve_fit(lambda x, A, sigma: gauss(x, A, mu1, sigma), xs[r_to_slice(roione, fullroi)], hlog[r_to_slice(roione, fullroi)], p0=initpar)
#     # Finding one photon peak
#     mu2 = xs[(hlog - gauss(xs, p1[0], mu1, p1[1]))[r_to_slice(roione, fullroi)].argmax() + roione[0] - fullroi[0]]
#     return mu2 - mu1

# def relativegain(adus, pid, dig_gains, gainlevels, offsets, badpixels, roi=[0, 60]):
#     """
#     Return relative gain value based on histogram of High gain adus divided by Medium gain adus.

#     adus - agipd raw adu values of one memory cell (Pulse ID = pid)
#     pid - Pulse ID
#     dig_gains - agipd digital gain values
#     gainlevels - calibration gain levels
#     offsets - callibration ADU offsets
#     badpixels - bad pixels mask
#     """
#     # memory cell id
#     mid = pid // 4
#     # gain mode mask
#     gainmode = np.array([frame > gainlevels[1, mid] for frame in dig_gains], dtype=int)
#     # high gain adus
#     highgains = np.where((gainmode == 0) & (badpixels[0, mid] == 0), adus - offsets[0, mid], 0)
#     highgains[highgains < 0] = 0
#     # medium gain adus
#     midgains = np.where((gainmode == 1) & (badpixels[1, mid] == 0), adus - offsets[1, mid], 0)
#     midgains[midgains < 0] = 0
#     # Finding mean ADU value for high gain adus
#     htotals = np.sum(np.array(highgains != 0, dtype=int), axis=0)
#     avhgains = np.where(htotals != 0, highgains.sum(axis=0) / htotals, 0)
#     # Finding mean ADU value for high gain adus
#     mtotals = np.sum(np.array(midgains != 0, dtype=int), axis=0)
#     avmgains = np.where(mtotals != 0, midgains.sum(axis=0) / mtotals, 0)
#     # Division of mean high gain adus and medium gain adus gives us relative gain value
#     relgains = np.where((avhgains != 0) & (avmgains != 0), avhgains / avmgains, 0)
#     # Finding relative gain value by fitting gaussian to histgram
#     hist, bin_edges = np.histogram(relgains.ravel(), 100, range=(0.5, 100.5))
#     bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
#     p, _ = curve_fit(lambda x, mu, sigma: gauss(x, hist.max(), mu, sigma), bin_centres[roi[0]:roi[1]], hist[roi[0]:roi[1]], bounds=([roi[0], 0], [roi[1], 10]))
#     return p[0]