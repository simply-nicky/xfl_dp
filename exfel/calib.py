"""
calib.py - calibration module
"""
import sys
import h5py
import numpy as np
import pyqtgraph as pg
from scipy.optimize import curve_fit
from . import utils

try:
    from PyQt5 import QtCore, QtGui, QtWidgets
except ImportError:
    from PyQt4 import QtCore, QtGui, QtWidgets

MODULES_NUM = 16
HG_GAIN = 1 / 68.8
MG_GAIN = 1 / 1.376
LG_GAIN = 5 / 1.376
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

FULL_ROI = ROI(-50, 150)
ZERO_ROI = ROI(-50, 50)
ONE_ROI = ROI(30, 100)
REL_ROI = ROI(20, 60)

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
                                                self.zero_adu,
                                                self.zero_fit[1]),
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
                                               self.one_adu,
                                               self.one_fit[1]),
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
        self.zero_adu = self.adus[self.hist[zero_slice].argmax()]
        self.zero_fit, _ = curve_fit(lambda x, amplitude, sigma: gauss(x,
                                                                       amplitude,
                                                                       self.zero_adu,
                                                                       sigma),
                                     self.adus[zero_slice],
                                     self.hist[zero_slice])
        one_slice = self.one_roi.relative_roi(self.full_roi).index
        self.one_hist = (self.hist - gauss(self.adus,
                                           self.zero_fit[0],
                                           self.zero_adu,
                                           self.zero_fit[1]))
        self.one_adu = self.adus[one_slice][self.one_hist[one_slice].argmax()]
        self.one_fit, _ = curve_fit(lambda x, amplitude, sigma: gauss(x,
                                                                      amplitude,
                                                                      self.one_adu,
                                                                      sigma),
                                    self.adus[one_slice],
                                    self.one_hist[one_slice])

    def update_plot(self):
        self.update_fit()
        self.zero_plot.setData(self.adus, gauss(self.adus,
                                                self.zero_fit[0],
                                                self.zero_adu,
                                                self.zero_fit[1]))
        self.one_plot.setData(self.adus, gauss(self.adus,
                                               self.one_fit[0],
                                               self.one_adu,
                                               self.one_fit[1]))
        self.zero_label.setText("Zero ADU: {:5.1f}".format(self.zero_adu))
        self.one_label.setText("One ADU: {:5.1f}".format(self.one_adu))

def run_app(hist, adus):
    app = QtCore.QCoreApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    main_win = CalibViewer(hist, adus)
    app.exec_()
    return main_win.zero_adu, main_win.one_adu

class DarkCalib(object):
    OFFSET_KEY = OFFSET_KEY
    BADMASK_KEY = BADMASK_KEY
    GAIN_LEVEL_KEY = GAIN_LEVEL_KEY

    def __init__(self, filename):
        self.data_file = h5py.File(filename, 'r')

    def offset(self, gain_mode, pid):
        return self.data_file[self.OFFSET_KEY][gain_mode, pid]

    def bad_mask(self, gain_mode, pid):
        return self.data_file[self.BADMASK_KEY][gain_mode, pid]

    def bad_mask_inv(self, gain_mode, pid):
        return 1 - self.bad_mask(gain_mode, pid)

    def gain_level(self, gain_mode, pid):
        return self.data_file[self.GAIN_LEVEL_KEY][gain_mode, pid]

class CalibData(object):
    def __init__(self, data, dark_calib):
        self.raw_data, self.raw_gain = data[utils.DATA_KEY], data[utils.GAIN_KEY]
        self.pid, self.train_id = data[utils.PULSE_KEY][0], data[utils.TRAIN_KEY]
        self.hg_calib = dark_calib.calib_data(utils.HIGH_GAIN, self.pid)
        self.mg_calib = dark_calib.calib_data(utils.MEDIUM_GAIN, self.pid)

    @property
    def hg_mask(self):
        gain_mask = self.raw_gain < self.mg_calib.gain_level
        return gain_mask & self.hg_calib.bad_mask_inv

    @property
    def mg_mask(self):
        gain_mask = self.raw_gain > self.mg_calib.gain_level
        return gain_mask & self.mg_calib.bad_mask_inv

    @property
    def hg_data(self):
        return (self.raw_data - self.hg_calib.offset)[self.hg_mask]

    @property
    def mg_data(self):
        return (self.raw_data - self.mg_calib.offset)[self.mg_mask]

    def hg_histogram(self, roi=(-50, 150)):
        roi_obj = ROI(roi[0], roi[1])
        hist, hg_edges = np.histogram(self.hg_data, roi_obj.length, range=roi_obj.bounds)
        zero_peak = abs(roi_obj.lower_bound)
        hist[zero_peak] = (hist[zero_peak + 1] + hist[zero_peak - 1]) / 2
        log_hist, adus = np.log(hist) - np.log(hist).min(), (hg_edges[1:] + hg_edges[:-1]) / 2
        return log_hist, adus

    def hg_calibrate_gui(self, roi=(-50, 150)):
        hist, adus = self.hg_histogram(roi)
        zero_adu, one_adu = run_app(hist, adus)
        return zero_adu, one_adu

    def hg_calibrate(self, full_roi=(-50, 150), zero_roi=(-50, 50), one_roi=(30, 100)):
        hist, adus = self.hg_histogram(full_roi)
        zero_slice = slice(int(zero_roi[0] - full_roi[0]), int(zero_roi[1] - full_roi[0]))
        zero_adu = adus[hist[zero_slice].argmax()]
        zero_fit, _ = curve_fit(lambda x, amplitude, sigma: gauss(x, amplitude, zero_adu, sigma),
                                adus[zero_slice],
                                hist[zero_slice])
        one_slice = slice(int(one_roi[0] - full_roi[0]), int(one_roi[1] - full_roi[0]))
        one_adu = adus[one_slice][(hist - gauss(adus, zero_fit[0], zero_adu, zero_fit[1]))[one_slice].argmax()]
        return zero_adu, one_adu

    def mg_calibrate(self, rel_roi=(20, 60)):
        hg_totals = np.sum(np.array(self.hg_data <= 0, dtype=np.uint32), axis=0)
        mg_totals = np.sum(np.array(self.mg_data <= 0, dtype=np.uint32), axis=0)
        hg_average = np.where(hg_totals != 0, self.hg_data.sum(axis=0) / hg_totals, 0)
        mg_average = np.where(mg_totals != 0, self.mg_data.sum(axis=0) / mg_totals, 0)
        rel_data = np.where(mg_average != 0, hg_average / mg_average, 0)
        rel_hist, edges = np.histogram(rel_data.ravel(), rel_roi[1] - rel_roi[0], range=rel_roi)
        adus = (edges[:-1] + edges[1:]) / 2
        rel_fit, _ = curve_fit(lambda x, mu, sigma: gauss(x, rel_hist.max(), mu, sigma),
                               adus,
                               rel_hist,
                               bounds=([rel_roi[0], 0], [rel_roi[1], 20]))
        return rel_fit[0]