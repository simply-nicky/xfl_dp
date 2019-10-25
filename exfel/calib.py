"""
calib.py - calibration module
"""
import sys
import concurrent.futures
import h5py
import os
import numpy as np
import pyqtgraph as pg
from scipy.optimize import curve_fit
from scipy.ndimage.filters import median_filter
from .utils import HIGH_GAIN, MEDIUM_GAIN, make_output_dir

try:
    from PyQt5 import QtCore, QtGui, QtWidgets
except ImportError:
    from PyQt4 import QtCore, QtGui, QtWidgets

HG_GAIN = 1 / 68.8
MG_GAIN = 1 / 1.376
CELL_ID = 1

OFFSET_KEY = "AnalogOffset"
BADMASK_KEY = "Badpixel"
GAIN_LEVEL_KEY = "DigitalGainLevel"

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
    def bounds(self):
        return (self.lower_bound, self.higher_bound)

    @property
    def index(self):
        return slice(int(self.lower_bound), int(self.higher_bound))

    def relative_roi(self, base_roi):
        return ROI(self.lower_bound - base_roi.lower_bound,
                   self.higher_bound - base_roi.lower_bound)

class CalibViewer(QtGui.QWidget):
    def __init__(self, hist, adus, parent=None):
        super(CalibViewer, self).__init__(parent=parent)
        self.hist, self.adus = hist, adus
        self.full_roi = ROI(self.adus.min(), self.adus.max())
        self.zero_roi = ROI(self.full_roi.lower_bound,
                            self.full_roi.lower_bound + 0.4 * self.full_roi.length)
        self.one_roi = ROI(self.full_roi.higher_bound - 0.55 * self.full_roi.length,
                           self.full_roi.higher_bound - 0.05 * self.full_roi.length)
        self.update_fit()
        self.init_ui()

    @property
    def zero_adu(self):
        return self.zero_fit[1]

    @property
    def one_adu(self):
        return self.one_fit[1]

    def init_ui(self):
        self.vbox_layout = QtGui.QVBoxLayout()
        label_widget = QtGui.QLabel("ADU Histogram")
        label_widget.setFont(QtGui.QFont('SansSerif', 20))
        self.vbox_layout.addWidget(label_widget)
        plot_widget = pg.PlotWidget(name="Plot", background='w')
        one_hist_plot = plot_widget.plot(self.adus, self.one_hist, atialias=True)
        one_hist_plot.setPen(color=(255, 0, 0, 150), width=2, style=QtCore.Qt.DashDotLine)
        hist_plot = plot_widget.plot(self.adus, self.hist, anitalias=True)
        hist_plot.setPen(color=(0, 0, 0, 255), width=3)
        self.zero_plot = plot_widget.plot(self.adus,
                                          gauss(self.adus,
                                                self.zero_fit[0],
                                                self.zero_fit[1],
                                                self.zero_fit[2]),
                                          antialias=True)
        self.zero_plot.setPen(color='b', width=2, style=QtCore.Qt.DashLine)
        self.zero_lr = pg.LinearRegionItem(values=list(self.zero_roi.bounds),
                                           bounds=list(self.full_roi.bounds))
        self.zero_lr.setBrush(QtGui.QBrush(QtGui.QColor(0, 0, 255, 50)))
        self.zero_lr.sigRegionChanged.connect(self.update_zero_roi)
        plot_widget.addItem(self.zero_lr)
        self.one_plot = plot_widget.plot(self.adus,
                                         gauss(self.adus,
                                               self.one_fit[0],
                                               self.one_fit[1],
                                               self.one_fit[2]),
                                         antialias=True)
        self.one_plot.setPen(color='r', width=2, style=QtCore.Qt.DashLine)
        self.one_lr = pg.LinearRegionItem(values=list(self.one_roi.bounds),
                                          bounds=list(self.full_roi.bounds))
        self.one_lr.setBrush(QtGui.QBrush(QtGui.QColor(255, 0, 0, 50)))
        self.one_lr.sigRegionChanged.connect(self.update_one_roi)
        plot_widget.addItem(self.one_lr)
        self.vbox_layout.addWidget(plot_widget)
        hbox = QtGui.QHBoxLayout()
        update_button = QtGui.QPushButton("Update Plot")
        update_button.clicked.connect(self.update_plot)
        hbox.addWidget(update_button)
        exit_button = QtGui.QPushButton("Done")
        exit_button.clicked.connect(self.close)
        hbox.addWidget(exit_button)
        hbox.addStretch(1)
        self.zero_label = QtGui.QLabel("Zero ADU: {:5.1f}".format(self.zero_adu))
        hbox.addWidget(self.zero_label)
        self.one_label = QtGui.QLabel("One ADU: {:5.1f}".format(self.one_adu))
        hbox.addWidget(self.one_label)
        self.vbox_layout.addLayout(hbox)
        self.setLayout(self.vbox_layout)
        self.setGeometry(0, 0, 1280, 720)
        self.setWindowTitle('Photon Calibration')
        self.show()

    def update_zero_roi(self):
        self.zero_roi = ROI(*self.zero_lr.getRegion())

    def update_one_roi(self):
        self.one_roi = ROI(*self.one_lr.getRegion())

    def update_fit(self):
        zero_slice = self.zero_roi.relative_roi(self.full_roi).index
        zero_max = self.adus[zero_slice][self.hist[zero_slice].argmax()]
        self.zero_fit, _ = curve_fit(gauss,
                                     self.adus[zero_slice],
                                     self.hist[zero_slice],
                                     bounds=([0, zero_max - 10, 0], [np.inf, zero_max + 10, np.inf]))
        one_slice = self.one_roi.relative_roi(self.full_roi).index
        self.one_hist = (self.hist - gauss(self.adus,
                                           self.zero_fit[0],
                                           self.zero_fit[1],
                                           self.zero_fit[2]))
        one_max = self.adus[one_slice][self.one_hist[one_slice].argmax()]
        self.one_fit, _ = curve_fit(gauss,
                                    self.adus[one_slice],
                                    self.one_hist[one_slice],
                                    bounds=([0, one_max - 10, 0], [np.inf, one_max + 10, np.inf]))

    def update_plot(self):
        self.update_fit()
        self.zero_plot.setData(self.adus, gauss(self.adus,
                                                self.zero_fit[0],
                                                self.zero_fit[1],
                                                self.zero_fit[2]))
        self.one_plot.setData(self.adus, gauss(self.adus,
                                               self.one_fit[0],
                                               self.one_fit[1],
                                               self.one_fit[2]))
        self.zero_label.setText("Zero ADU: {:5.1f}".format(self.zero_adu))
        self.one_label.setText("One ADU: {:5.1f}".format(self.one_adu))

def run_app(hist, adus):
    app = QtCore.QCoreApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    main_win = CalibViewer(hist, adus)
    app.exec_()
    return main_win.zero_adu, main_win.one_adu

class DarkAGIPD(object):
    OFFSET_KEY = OFFSET_KEY
    BADMASK_KEY = BADMASK_KEY
    GAIN_LEVEL_KEY = GAIN_LEVEL_KEY
    MODULE_SHAPE = (512, 128)

    def __init__(self, filename):
        self.data_file = h5py.File(filename, 'r')

    def offset(self, gain_mode, cell_id, module_id):
        return self.data_file[self.OFFSET_KEY][gain_mode, cell_id, module_id]

    def gain_level(self, gain_mode, cell_id, module_id):
        return self.data_file[self.GAIN_LEVEL_KEY][gain_mode, cell_id, module_id]

    def bad_mask(self, module_id):
        idxs = (slice(self.MODULE_SHAPE[0] * module_id, self.MODULE_SHAPE[0] * (module_id + 1)),)
        return self.data_file[self.BADMASK_KEY][idxs]

    def bad_mask_inv(self, module_id):
        return 1 - self.bad_mask(module_id)

class AGIPDCalib(object):
    GAIN = np.array([HG_GAIN, MG_GAIN])
    CELL_ID = CELL_ID
    FLAT_ROI = (0, 10)

    def __init__(self, raw_data, raw_gain, dark, module_id, mask_inv=True):
        self.raw_data, self.raw_gain, self.dark, self.module_id = raw_data, raw_gain, dark, module_id
        self.bad_mask = self.dark.bad_mask_inv(module_id) if mask_inv else self.dark.bad_mask(module_id)
        self._init_adu()
        self._flat_correct()
        self._init_mask()
        self.data = ((self.adu * self.mask).T * self.GAIN).T

    def _init_adu(self):
        hg_adus = self.raw_data - self.dark.offset(HIGH_GAIN, self.CELL_ID, self.module_id)
        mg_adus = self.raw_data - self.dark.offset(MEDIUM_GAIN, self.CELL_ID, self.module_id)
        self.adu = np.stack((hg_adus, mg_adus))

    def _flat_correct(self):
        self.zero_levels = self.adu[0, :, self.FLAT_ROI[0]:self.FLAT_ROI[1]].mean(axis=(1, 2))
        self.adu[0] = (self.adu[0].T - self.zero_levels).T

    def _init_mask(self):
        hg_mask = (self.raw_gain < self.dark.gain_level(MEDIUM_GAIN,
                                                        self.CELL_ID,
                                                        self.module_id)).astype(np.uint8)
        mg_mask = (self.raw_gain > self.dark.gain_level(MEDIUM_GAIN,
                                                        self.CELL_ID,
                                                        self.module_id)).astype(np.uint8)
        self.mask = np.stack((hg_mask, mg_mask)) * self.bad_mask

    @property
    def calib_data(self):
        return self.data.sum(axis=0)

    def save_data(self, out_file):
        data_group = out_file.create_group('MODULE{:02d}'.format(self.module_id))
        data_group.create_dataset('adu', data=self.adu)
        data_group.create_dataset('mask', data=self.mask)
        data_group.create_dataset('data', data=self.data)

class HGData(object):
    ZERO_VERGE = 50

    def __init__(self, data):
        self.data = data
        self.optimize()

    @property
    def size(self):
        return self.data.shape[0]

    def hist_frame(self, idx, roi=(-200, 100)):
        hist, edges = np.histogram(self.data[idx].ravel(), roi[1] - roi[0], range=roi)
        if roi[0] < 0 < roi[1]:
            zero_peak = abs(roi[0])
            hist[zero_peak] = (hist[zero_peak - 1] + hist[zero_peak + 1]) / 2
        return hist, (edges[:-1] + edges[1:]) / 2

    def zero_adu(self, idx):
        hist, adus = self.hist_frame(idx, roi=(self.data[idx].min(), self.ZERO_VERGE))
        return adus[hist.argmax()]

    def optimize(self):
        futures = []
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for idx in range(self.size):
                futures.append(executor.submit(self.zero_adu, idx))
        self.zero_adus = np.array([fut.result() for fut in futures])
        self.data = (self.data.T - self.zero_adus).T

    def histogram(self, roi=(-100, 200)):
        return self.hist_frame(slice(0, self.size), roi=roi)

    def log_hist(self, roi=(-100, 200)):
        hist, adus = self.histogram(roi)
        hist[hist == 0] = 1
        return np.log(hist) - np.log(hist.min()), adus

    def calibrate_gui(self, roi=(-100, 200)):
        hist, adus = self.log_hist(roi)
        zero_adu, one_adu = run_app(median_filter(hist, 3), adus)
        return zero_adu, one_adu

    def calibrate(self, full_roi=(-100, 200), zero_roi=(-50, 50), one_roi=(30, 100)):
        hist, adus = self.log_hist(full_roi)
        hist = median_filter(hist, 3)
        zero_slice = slice(int(zero_roi[0] - full_roi[0]), int(zero_roi[1] - full_roi[0]))
        zero_max = adus[zero_slice][hist[zero_slice].argmax()]
        zero_fit, _ = curve_fit(gauss,
                                adus[zero_slice],
                                hist[zero_slice],
                                bounds=([0, zero_max - 10, 0], [np.inf, zero_max + 10, np.inf]))
        one_slice = slice(int(one_roi[0] - full_roi[0]), int(one_roi[1] - full_roi[0]))
        one_hist = (hist - gauss(adus, zero_fit[0], zero_fit[1], zero_fit[2]))[one_slice]
        one_max = adus[one_slice][one_hist.argmax()]
        one_fit, _ = curve_fit(gauss,
                               adus[one_slice],
                               one_hist,
                               bounds=([0, one_max - 10, 0], [np.inf, one_max + 10, np.inf]))
        return zero_fit[1], one_fit[1]

    # def mg_calibrate(self, rel_roi=(20, 60)):
    #     hg_totals = np.sum(np.array(self.hg_data <= 0, dtype=np.uint32), axis=0)
    #     mg_totals = np.sum(np.array(self.mg_data <= 0, dtype=np.uint32), axis=0)
    #     hg_average = np.where(hg_totals != 0, self.hg_data.sum(axis=0) / hg_totals, 0)
    #     mg_average = np.where(mg_totals != 0, self.mg_data.sum(axis=0) / mg_totals, 0)
    #     rel_data = np.where(mg_average != 0, hg_average / mg_average, 0)
    #     rel_hist, edges = np.histogram(rel_data.ravel(), rel_roi[1] - rel_roi[0], range=rel_roi)
    #     adus = (edges[:-1] + edges[1:]) / 2
    #     rel_fit, _ = curve_fit(lambda x, mu, sigma: gauss(x, rel_hist.max(), mu, sigma),
    #                            adus,
    #                            rel_hist,
    #                            bounds=([rel_roi[0], 0], [rel_roi[1], 20]))
    #     return rel_fit[0]
