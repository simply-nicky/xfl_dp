"""
calib.py - calibration module
"""
import sys
import numpy as np
import pyqtgraph as pg
from scipy.optimize import curve_fit
from . import utils

try:
    from PyQt5 import QtCore, QtGui, QtWidgets
except ImportError:
    from PyQt4 import QtCore, QtGui, QtWidgets

MODULE_SHAPE = (512, 128)
AGIPD_SHAPE = (8192, 128)
MODULES_NUM = 16
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
    def bounds(self):
        return (self.lower_bound, self.higher_bound)

    @property
    def index(self):
        return slice(int(self.lower_bound), int(self.higher_bound))

    def relative_roi(self, base_roi):
        return ROI(self.lower_bound - base_roi.lower_bound,
                   self.higher_bound - base_roi.lower_bound)

FULL_ROI = ROI(-50, 150)
ZERO_ROI = ROI(-50, 30).relative_roi(FULL_ROI)
ONE_ROI = ROI(30, 150).relative_roi(FULL_ROI)
REL_ROI = ROI(20, 60)

class CalibParameter(object):
    def __init__(self, level):
        if isinstance(level, (int, float)):
            self.agipd = level * np.ones(AGIPD_SHAPE)
        elif isinstance(level, np.ndarray) and level.shape == MODULE_SHAPE:
            self.agipd = np.tile(level, (MODULES_NUM, 1))
        elif isinstance(level, np.ndarray) and level.shape == AGIPD_SHAPE:
            self.agipd = level
        else:
            raise ValueError('Invalid level value')

    def module(self, module_id):
        return self.agipd[module_id*MODULE_SHAPE[0]:(module_id+1)*MODULE_SHAPE[0]]

class GainLevel(object):
    def __init__(self, hg_level=HG_LEVEL, mg_level=MG_LEVEL, lg_level=LG_LEVEL):
        self.hg_level = CalibParameter(hg_level)
        self.mg_level = CalibParameter(mg_level)
        self.lg_level = CalibParameter(lg_level)

    def mask_module(self, data, module_id):
        mg_mask = (data[utils.GAIN_KEY] > self.mg_level.module(module_id)).astype(np.uint8)
        lg_mask = (data[utils.GAIN_KEY] > self.lg_level.module(module_id)).astype(np.uint8)
        return mg_mask + lg_mask

    def mask_agipd(self, data):
        mg_mask = (data[utils.GAIN_KEY] > self.mg_level.agipd).astype(np.uint8)
        lg_mask = (data[utils.GAIN_KEY] > self.lg_level.agipd).astype(np.uint8)
        return mg_mask + lg_mask

    def get_module_data(self, data, module_id, gain_mode):
        mask = self.mask_module(data, module_id)
        return data[utils.DATA_KEY][mask == gain_mode]

    def get_agipd_data(self, data, gain_mode):
        mask = self.mask_agipd(data)
        return data[utils.DATA_KEY][mask == gain_mode]

class CalibViewer(QtGui.QWidget):
    def __init__(self, hist, adus, filename, parent=None):
        super(CalibViewer, self).__init__(parent=parent)
        self.hist, self.adus, self.filename = hist, adus, filename
        self.full_roi = ROI(adus.min(), adus.max())
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
        self.setWindowTitle('Photon Calibration, {}'.format(self.filename))
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

def run_app(hist, adus, filename):
    app = QtCore.QCoreApplication.instance()
    if app is None:
        app = QtWidgets.QApplication(sys.argv)
    main_win = CalibViewer(hist, adus, filename)
    app.exec_()
    return main_win.zero_adu, main_win.one_adu

def hg_calibrate(hg_data, full_roi=FULL_ROI, zero_roi=ZERO_ROI, one_roi=ONE_ROI):
    # making high gain histogram
    hist, hg_edges = np.histogram(hg_data.ravel(), full_roi.length, range=full_roi.bounds)
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

def hg_gui_calibrate(hg_data, filename, full_roi=FULL_ROI):
    hist, hg_edges = np.histogram(hg_data.ravel(), full_roi.length, range=full_roi.bounds)
    # Supressing 0 ADU value peak
    zero_peak = abs(full_roi.lower_bound)
    hist[zero_peak] = (hist[zero_peak + 1] + hist[zero_peak - 1]) / 2
    # making log histogram
    log_hist, adus = np.log(hist) - np.log(hist).min(), (hg_edges[1:] + hg_edges[:-1]) / 2
    zero_adu, one_adu = run_app(log_hist, adus, filename)
    return zero_adu, one_adu

def mg_calibrate(mg_data, hg_data, rel_roi=REL_ROI):
    hg_totals = np.sum(np.array(hg_data <= 0, dtype=np.uint32), axis=0)
    mg_totals = np.sum(np.array(mg_data <= 0, dtype=np.uint32), axis=0)
    hg_average = np.where(hg_totals != 0, hg_data.sum(axis=0) / hg_totals, 0)
    mg_average = np.where(mg_totals != 0, mg_data.sum(axis=0) / mg_totals, 0)
    rel_data = np.where(mg_average != 0, hg_average / mg_average, 0)
    rel_hist, edges = np.histogram(rel_data.ravel(), rel_roi.length, range=rel_roi.bounds)
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