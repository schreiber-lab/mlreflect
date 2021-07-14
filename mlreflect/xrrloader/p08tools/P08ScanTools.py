# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 17:18:21 2018

@author: Florian
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import sys
import traceback
from os import listdir

import fabio
import h5py
import matplotlib.pyplot as plt
import numpy
from scipy import optimize

from .fio_reader import read


class Scan(object):

    def __init__(self):
        self.filename = None

        self.scan_motor_names = []
        self.scan_motors = []

        self.scan_cmd = None
        self.header_info = None

        self.is_lambda = False
        self.is_p100k = False
        self.is_p300k = False
        self.is_eiger = False
        self.is_mythen = False
        self.is_pe = False

        self.detectors = []

        self.image_data = {}

        self.remove_idxes = None

    def load_scan(self, filename=None, auto_load_images=True, auto_remove_double=False):

        self.scan_motor_names = []
        self.scan_motors = []

        self.is_lambda = False
        self.is_p100k = False
        self.is_p300k = False
        self.is_eiger = False
        self.is_mythen = False
        self.is_pe = False

        self.doubles_removed = False

        self.detectors = []
        self.image_data = {}

        if not filename is None:
            self.filename = filename

        if self.filename is None:
            raise Exception("filename is not defined")

        self.remove_idxes = None

        motor_positions, column_names, data, header_info = read(self.filename)

        self.scan_name = self.filename.split('/')[-1]
        self.scan_name = self.scan_name.rsplit('.', 1)[0]
        self.scan_dir = self.filename.rsplit('/', 1)[0]

        self.scan_cmd = header_info["scan_cmd"]
        self.motor_positions = motor_positions
        self.header_info = header_info
        self.data = data

        # extract rois from header_info
        if "rois" in self.header_info:
            self.rois = self.header_info["rois"]
        else:
            self.rois = None
        # extract UB matrix from header_info
        if "ubmatrix" in self.header_info:
            self.ubmatrix = self.header_info["ubmatrix"]
        else:
            self.ubmatrix = None

        # extract motor names:
        if not self.scan_cmd.find('lscan') == -1 and self.scan_cmd.find('hklscan') == -1:
            self.scan_motor_names = ['l']
            self.scan_motors = [numpy.array(data["kozhue6cctrl_l"])]
        elif not self.scan_cmd.find('kscan') == -1:
            self.scan_motor_names = ['k']
            self.scan_motors = [numpy.array(data["kozhue6cctrl_k"])]
        elif not self.scan_cmd.find('hscan') == -1:
            self.scan_motor_names = ['h']
            self.scan_motors = [numpy.array(data["kozhue6cctrl_h"])]
        elif not self.scan_cmd.find('hklscan') == -1:
            self.scan_motor_names = ['h', 'k', 'l']
            self.scan_motors = [numpy.array(data["kozhue6cctrl_h"]), numpy.array(data["kozhue6cctrl_k"]),
                                numpy.array(data["kozhue6cctrl_l"])]
        elif not self.scan_cmd.find('ascan') == -1 or not self.scan_cmd.find('dscan') == -1:
            mot1_name = self.scan_cmd.split()[1]
            self.scan_motor_names = [mot1_name]
            self.scan_motors = [numpy.array(data[mot1_name])]
        elif not self.scan_cmd.find('a2scan') == -1 or not self.scan_cmd.find('d2scan') == -1:
            mot1_name = self.scan_cmd.split()[1]
            mot2_name = self.scan_cmd.split()[4]
            self.scan_motor_names = [mot1_name, mot2_name]
            self.scan_motors = [numpy.array(data[mot1_name]), numpy.array(data[mot2_name])]
        elif not self.scan_cmd.find('a3scan') == -1 or not self.scan_cmd.find('d3scan') == -1:
            mot1_name = self.scan_cmd.split()[1]
            mot2_name = self.scan_cmd.split()[4]
            mot3_name = self.scan_cmd.split()[7]
            self.scan_motor_names = [mot1_name, mot2_name, mot3_name]
            self.scan_motors = [numpy.array(data[mot1_name]), numpy.array(data[mot2_name]),
                                numpy.array(data[mot3_name])]

        # check which 1D/2D detectors were used during the scan
        self.detectors = []

        for col_name in column_names:
            if not col_name.find('lambda') == -1:
                self.is_lambda = True
            elif not col_name.find('p100k') == -1:
                self.is_p100k = True
            elif not col_name.find('p300k') == -1:
                self.is_p300k = True
            elif not col_name.find('eiger') == -1:
                self.is_eiger = True
            elif not (col_name.find('mythint') == -1 and col_name.find('mythmax') == -1 and col_name.find(
                    'mythroi1') == -1 and col_name.find('mythroi2') == -1):
                self.is_mythen = True
            elif not col_name.find('pe_roi') == -1 or not col_name.find('pe_trigger') == -1:
                self.is_pe = True

        if self.is_lambda:
            self.detectors.append('lambda')
        if self.is_p100k:
            self.detectors.append('p100k')
        if self.is_p300k:
            self.detectors.append('p300k')
        if self.is_eiger:
            self.detectors.append('eiger')
        if self.is_mythen:
            self.detectors.append('mythen')
        if self.is_pe:
            self.detectors.append('pe')

        # load images if autoload is active
        if auto_load_images:
            self.load_image_stack()

        # remove doubled data points if autoremove is active
        if auto_remove_double:
            self.remove_double()

    def load_image_stack(self):
        '''
        loads the detector files for the detectors used in the scan.
        data is stored in the image_data dictionary
        '''

        if self.is_lambda:
            lambda_data = self._load_lambda()
            self.image_data["lambda"] = self._remove_double_fromstack(lambda_data)
        if self.is_p100k:
            p100k_data = self._load_pilatus("p100k")
            self.image_data["p100k"] = self._remove_double_fromstack(p100k_data)
        if self.is_eiger:
            eiger_data = self._load_eiger()
            self.image_data["eiger"] = self._remove_double_fromstack(eiger_data)
        if self.is_mythen:
            mythen_data = self._load_mythen("mythen")
            self.image_data["mythen"] = mythen_data

    def _remove_double_fromstack(self, stack):
        '''
        remove doubled data points from an image stack if remove_idxes is filled.
        i.e. if remove_double has already been used.
        This should only be used internaly if image files are loaded AFTER remove_double
        has been executed
        '''
        if not self.remove_idxes is None:
            return numpy.delete(stack, self.remove_idxes, axis=0)
        else:
            return stack

    def _load_eiger(self):
        '''
        load h5 file from a eiger1m detector und return a numpy 3D array with the data.

        If an index column for the detector is used, only files with corresponding index are load.
        i.e. data index and image index are matching

        Currently, only one h5 file per scan is supported!
        '''

        eiger_folder = "eiger1m"

        print("ScanName: %s" % self.scan_name)

        if len(self.scan_name.split('.')) == 1:
            folder_name = self.scan_name
        else:
            folder_part1 = self.scan_name.split('.')[0]
            folder_part2 = self.scan_name.split('.')[1].rsplit('_', 1)[1]
            folder_name = "%s_%s" % (folder_part1, folder_part2)

        eiger_file_path = "%s/%s/%s" % (self.scan_dir, folder_name, eiger_folder)

        self.eiger_file = "%s/%s_master.h5" % (eiger_file_path, folder_name)

        print("eiger_file: %s" % self.eiger_file)

        try:
            nxsfile = h5py.File(self.eiger_file, 'r')
        except:
            print("could not load hdf5 file. search for master file")

            files = listdir(eiger_file_path)

            for filename in files:
                if "_master.h5" in filename:
                    self.eiger_file = "%s/%s" % (eiger_file_path, filename)
                    nxsfile = h5py.File(self.eiger_file, 'r')
                    break

        nxsdata = nxsfile.get('/entry/data/data_000001')

        # print( nxsdata.shape )

        eiger_idx = None
        try:
            eiger_idx = self.data["eiger_index"]
        except:
            print("no index column for eiger available")

        if not eiger_idx is None:
            nr_images_in_stack = nxsdata.shape[0]
            remove_idxes = []
            for idx in range(nr_images_in_stack):
                if not idx in eiger_idx:
                    remove_idxes.append(idx)

            nxsdata = numpy.delete(nxsdata, remove_idxes, axis=0)
            # print( nxsdata.shape )

        return nxsdata

    def _load_lambda(self):
        '''
        load nxs file from a lambda detector und return a numpy 3D array with the data.

        If an index column for the detector is used, only files with corresponding index are load.
        i.e. data index and image index are matching

        Currently, only one nxs file per scan is supported!
        '''

        self.lambda_file = "%s/%s/lambda/%s_00000.nxs" % (self.scan_dir, self.scan_name, self.scan_name)
        # print ( "lambda_file: %s" % self.lambda_file)

        nxsfile = h5py.File(self.lambda_file)
        nxsdata = nxsfile.get('/entry/instrument/detector/data')

        # print( nxsdata.shape )

        lambda_idx = None
        try:
            lambda_idx = self.data["lambda_index"]
        except:
            print("no index column for lambda available")

        if not lambda_idx is None:
            nr_images_in_stack = nxsdata.shape[0]
            remove_idxes = []
            for idx in range(nr_images_in_stack):
                if not idx in lambda_idx:
                    remove_idxes.append(idx)

            nxsdata = numpy.delete(nxsdata, remove_idxes, axis=0)
            # print( nxsdata.shape )

        return nxsdata

    def _load_pilatus(self, detector):
        '''
        load tif or cbf files from a pilatus detector und return a numpy 3D array with the data.

        If an index column for the detector is used, only files with corresponding index are load.
        i.e. data index and image index are matching
        '''

        self.pilatus_dir = "%s/%s/%s/" % (self.scan_dir, self.scan_name, detector)

        file_list = glob.glob("%s%s*.tif" % (self.pilatus_dir, self.scan_name))
        file_list.extend(glob.glob("%s%s*.cbf" % (self.pilatus_dir, self.scan_name)))
        file_list = sorted(file_list)

        # print (file_list)
        nr_files = len(file_list)

        pilatus_idx = None
        try:
            pilatus_idx = self.data["%s_index" % detector]
            nr_files = len(pilatus_idx)
        except:
            print("no index column for %s available" % detector)

        firstdata = fabio.open(file_list[0]).data

        pilatus_data = numpy.zeros([nr_files, firstdata.shape[0], firstdata.shape[1]], dtype=firstdata.dtype)

        if not pilatus_idx is None:
            cnt = 0
            for idx in pilatus_idx:
                filename = "%s%s_%05d" % (self.pilatus_dir, self.scan_name, idx)
                if (filename + ".tif") in file_list:
                    pilatus_data[cnt, :, :] = fabio.open(filename + ".tif").data
                    cnt += 1
                elif (filename + ".cbf") in file_list:
                    pilatus_data[cnt, :, :] = fabio.open(filename + ".tif").data
                    cnt += 1

        # print( pilatus_data.shape )
        # print( len(pilatus_idx) )

        return pilatus_data

    def _load_mythen(self, detector):
        '''
        load .raw files from mythen detector
        '''

        self.mythen_dir = "%s/%s/%s/" % (self.scan_dir, self.scan_name, detector)

        file_list = glob.glob("%s%s*.raw" % (self.mythen_dir, self.scan_name))
        file_list = sorted(file_list)

        first_file = self._load_1D(file_list[0])

        # print (file_list)
        nr_files = len(file_list)

        image = numpy.zeros([nr_files, len(first_file)])

        cnt = 0
        for filename in sorted(file_list):
            image[cnt, :] = self._load_1D(filename)
            cnt += 1

        return image

    def _load_1D(self, filename, columns=2):

        data = numpy.loadtxt(filename)

        if columns == 2:
            return data[:, 1]

    def remove_double(self):

        if self.doubles_removed:
            return

        pos = self.scan_motors[0]

        remove_idxes = []

        last_pos = numpy.nan

        for idx in range(len(pos)):

            if idx > 0:
                if numpy.abs(pos[idx] - last_pos) == 0:
                    remove_idxes.append(idx - 1)

            last_pos = pos[idx]

        # print( "remove_idxes: %s" % repr(remove_idxes) )

        for key in self.data:
            self.data[key] = numpy.delete(self.data[key], remove_idxes)
            # print("removing from: %s (now: %d)" % (key, len(self.data[key]) ))

        for idx in range(len(self.scan_motors)):
            self.scan_motors[idx] = numpy.delete(self.scan_motors[idx], remove_idxes)

        self.remove_idxes = remove_idxes

        for key in self.image_data:
            self.image_data[key] = numpy.delete(self.image_data[key], remove_idxes, axis=0)
            print("stack size: %s" % repr(self.image_data[key].shape))

        self.doubles_removed = True

    def calc_q(self):

        energy = self.motor_positions["energyfmb"]

        wavelength = 12380 / energy

        k = 2 * numpy.pi / wavelength

        if 'tt' in self.scan_motor_names and 'om' in self.scan_motor_names:
            print('This is a theta2theta scan')

            for idx in range(len(self.scan_motor_names)):
                if self.scan_motor_names[idx] == 'tt':
                    break

            pos = self.scan_motors[idx]

            q = 2 * k * numpy.sin(pos / 2 * numpy.pi / 180)

            print("k = %f" % k)

            self.scan_motor_names.append('q')
            self.scan_motors.append(q)

    def __str__(self):

        s = "Scan Object" + "\n"
        s += "filename:    %s " % self.filename + "\n"
        s += "scan cmd:    %s " % self.scan_cmd + "\n"
        s += "scan motors: %s " % self.scan_motor_names + "\n"
        # s += "motor_positions: %s " % self.motor_positions + "\n"
        s += "header_info: %s " % self.header_info + "\n"
        s += "detectors: %s " % self.detectors + "\n"

        return s


class ScanAnalyzer(object):

    def __init__(self):
        pass

    @staticmethod
    def export2ascii(scan, detector, roi, filename, xcolName=None, bck=None):

        xcol_idx = 0

        if not xcolName is None:
            for idx in range(len(scan.scan_motor_names)):
                if scan.scan_motor_names[idx] == xcolName:
                    xcol_idx = idx
                    break

        result = ScanAnalyzer.extract_rois(scan, {"fit_roi": roi})
        roi_intensity = numpy.array(result[detector]["fit_roi"])

        if not bck is None:
            result = ScanAnalyzer.extract_rois(scan, {"bck_int": bck})
            bck_int = numpy.array(result[detector]["bck_int"])

            roi_intensity = roi_intensity - bck_int

        xcol = scan.scan_motors[xcol_idx]
        xcol_name = scan.scan_motor_names[xcol_idx]

        data = open(filename, 'w')
        for pos, inten in zip(xcol, roi_intensity):
            # print("%6.4f, %6.4f" % (pos, inten-bck_inten))
            data.write("%6.4f %f\n" % (pos, inten))
        data.close()

    @staticmethod
    def extract_rois(scan, rois):

        image_data = scan.image_data

        if "atten_position" in scan.data:
            atten = scan.data["atten_position"]
        else:
            atten = 1

        roi_intensities = {}

        for detector in image_data:

            img_stack = image_data[detector]

            if len(img_stack.shape) == 3:
                print("2D detector image stack")

                img_nr = img_stack.shape[0]

                print("imgs in stack: %d" % img_nr)

                roi_intensity = {}
                roi_intensity["all"] = []

                for key in rois:
                    roi_intensity[key] = []

                for idx in range(img_nr):

                    img = img_stack[idx]

                    key = "all"
                    roi_intensity[key].append(numpy.sum(img, axis=None))

                    for key in rois:
                        roi_img = img[rois[key][1]:rois[key][3], rois[key][0]:rois[key][2]]
                        roi_sum = numpy.sum(roi_img, axis=None)
                        roi_intensity[key].append(roi_sum)

                for key in roi_intensity:
                    roi_intensity[key] = numpy.array(roi_intensity[key]) * numpy.array(atten)
                roi_intensities[detector] = roi_intensity

            elif len(img_stack.shape) == 2:

                print("line detector image")

                img_nr = img_stack.shape[0]

                print("imgs in stack: %d" % img_nr)

                roi_intensity = {}
                roi_intensity["all"] = []

                for key in rois:
                    roi_intensity[key] = []

                for idx in range(img_nr):

                    img = img_stack[idx]

                    key = "all"
                    roi_intensity[key].append(numpy.sum(img, axis=None))

                    for key in rois:
                        roi_img = img[rois[key][0]:rois[key][1]]
                        roi_sum = numpy.sum(roi_img, axis=None)
                        roi_intensity[key].append(roi_sum)

                for key in roi_intensity:
                    roi_intensity[key] = numpy.array(roi_intensity[key]) * numpy.array(atten)
                roi_intensities[detector] = roi_intensity

        return roi_intensities

    @staticmethod
    def combine_scan_image_regions(scan_list, comb_rois):
        for scan in scan_list:
            print(scan)
            scan.remove_double()
        print(comb_rois)

        if len(scan_list[0].detectors) == 0:
            raise Exception("no 2D/1D detectors used in scan")

        detector = scan_list[0].detectors[0]

        detector_images = scan_list[0].image_data[detector]

        master_img = numpy.zeros(detector_images.shape)
        for idx in range(len(scan_list)):

            if "atten_position" in scan_list[idx].data:
                atten = numpy.array(scan_list[idx].data["atten_position"])
            else:
                atten = 1

            detector_images = scan_list[idx].image_data[detector]

            for idx2 in range(detector_images.shape[0]):
                detector_images[idx2] = atten[idx2] * detector_images[idx2]

                # atten_image = numpy.tensordot(atten,detector_images[:,comb_rois[idx][1]:comb_rois[idx][3], comb_rois[idx][0]:comb_rois[idx][2]], axes=([0],[0]))

            atten_image = detector_images[:, comb_rois[idx][1]:comb_rois[idx][3], comb_rois[idx][0]:comb_rois[idx][2]]

            master_img[:, comb_rois[idx][1]:comb_rois[idx][3], comb_rois[idx][0]:comb_rois[idx][2]] = atten_image

        print(master_img[0])

        combined_scan = Scan()

        combined_scan.scan_motors = scan_list[0].scan_motors
        combined_scan.scan_motor_names = scan_list[0].scan_motor_names
        combined_scan.detectors = [detector]
        combined_scan.image_data = {detector: master_img}
        combined_scan.doubles_removed = True

        combined_scan.data = {}

        return combined_scan

    @staticmethod
    def writeHDF(filename, scan):

        image_stack = scan.image_data[scan.detectors[0]]

        with h5py.File(filename, 'w') as f:
            dset = f.create_dataset("data", (image_stack.shape), dtype='float')

            for idx in range(image_stack.shape[0]):
                # print ("min: %f   max: %f" % (numpy.min(image_stack[idx,:,:]), numpy.max(image_stack[idx,:,:]) ) )
                dset[idx, :, :] = image_stack[idx, :, :]

        print("hdf writing complete")


class ScanFitter(object):

    def __init__(self):
        pass

    @staticmethod
    def simpleFit(scan, detector, roi, peaks):

        result = ScanAnalyzer.extract_rois(scan, {"fit_roi": roi})

        roi_intensity = numpy.array(result[detector]["fit_roi"])

        xcol = scan.scan_motors[0]
        xcol_name = scan.scan_motor_names[0]

        params = []

        peak_list = []

        for key in peaks:

            if peaks[key]["type"] == "gauss":
                params += peaks[key]["params"][0:3]

                peak_list.append(key)

        print(params)
        params += [peaks[key]["params"][3]]

        print(params)

        # fitfunc = lambda p, x: ScanFitter._gauss(x, p) # Target function
        # errfunc = lambda p, x, y: fitfunc(p, x) - y # Distance to the target function

        fitfunc = lambda p, x: ScanFitter._multi_gauss(x, p)  # Target function
        errfunc = lambda p, x, y: numpy.abs(
            numpy.log10(fitfunc(p, x)) - numpy.log10(y))  # Distance to the target function

        p1, success = optimize.leastsq(errfunc, params[:], args=(xcol, roi_intensity))

        print("p1 : " + repr(p1))

        fitfunc(params, xcol)

        # ydata = {'data' : roi_intensity, 'before_fit' : ScanFitter._multi_gauss(xcol, params), 'after_fit' : ScanFitter._multi_gauss(xcol, p1)  }
        # ydata = {'data' : roi_intensity, 'before_fit' : ScanFitter._multi_gauss(xcol, params) }

        ydata = {'data': roi_intensity, 'fit': ScanFitter._multi_gauss(xcol, p1)}

        for idx in range(len(peak_list)):
            name = peak_list[idx]

            print("% d: %s" % (idx, name))

            gauss_params = p1[idx * 3:idx * 3 + 3]

            gauss_params = numpy.append(gauss_params, p1[-1])

            print("gauss_params: " + repr(gauss_params))

            result = ScanFitter._gauss(xcol, gauss_params)

            ydata[name] = result

        ScanPlot.plotCurves(xcol, ydata)

    @staticmethod
    def gaussLorentzianFit(scan, detector, roi, peaks, wichtung_params=None):

        result = ScanAnalyzer.extract_rois(scan, {"fit_roi": roi})

        roi_intensity = numpy.array(result[detector]["fit_roi"])

        xcol = scan.scan_motors[0]
        xcol_name = scan.scan_motor_names[0]

        params = [0, 0, 0, 0, 0, 0, 0]

        peak_list = []

        if not len(peaks) == 2:
            raise Exception("online one gauss and one lorentian are alowed for gausLorentzianFit")

        gauss_loaded = False
        lorentzian_loaded = False

        for key in peaks:

            if peaks[key]["type"] == "gauss":

                if gauss_loaded:
                    raise Exception("online one gauss and one lorentian are alowed for gausLorentzianFit")

                params[3:6] = peaks[key]["params"][0:3]

                peak_list.append(key)

                gauss_loaded = True

            if peaks[key]["type"] == "lorentzian":

                if lorentzian_loaded:
                    raise Exception("online one gauss and one lorentian are alowed for gausLorentzianFit")

                params[0:3] = peaks[key]["params"][0:3]

                params[6] = peaks[key]["params"][3]

                peak_list.append(key)

                lorentzian_loaded = False

        print(params)

        if wichtung_params is None:
            wichtung_params = [0, 1, 0]

        wichtung = lambda p, x: p[2] * numpy.heaviside(x - p[0], 1) * numpy.heaviside(-x + p[1], 1) + 1

        fitfunc = lambda p, x: ScanFitter._gauss_lorentzian(x, p)  # Target function
        # errfunc = lambda p, x, y: numpy.abs(numpy.log10(fitfunc(p, x)) - numpy.log10(y)) # Distance to the target function

        errfunc = lambda p, x, y: wichtung(wichtung_params, x) * numpy.abs(
            numpy.log10(fitfunc(p, x) + 1) - numpy.log10(y + 1))  # Distance to the target function
        # errfunc = lambda p, x, y: numpy.abs( (fitfunc(p, x)+1) - (y+1) ) # Distance to the target function

        p1, success = optimize.leastsq(errfunc, params[:], args=(xcol, roi_intensity))

        # p1, success = optimize.curve_fit(fitfunc, xcol, roi_intensity)

        print("p1 : " + repr(p1))

        fitfunc(params, xcol)

        ydata = {'data': roi_intensity, 'fit': ScanFitter._gauss_lorentzian(xcol, p1),
                 'start': ScanFitter._gauss_lorentzian(xcol, params)}
        # ydata = {'data' : roi_intensity, 'start' : ScanFitter._gauss_lorentzian(xcol, params)  }
        # ydata = {'data' : roi_intensity, 'fit' : ScanFitter._gauss_lorentzian(xcol, p1)  }

        for idx in range(len(peak_list)):

            name = peak_list[idx]

            print("% d: %s" % (idx, name))

            if peaks[name]["type"] == "gauss":
                params = p1[3:6]
                params = numpy.append(params, p1[-1])
                print("params: " + repr(params))
                result = ScanFitter._gauss(xcol, params)
            elif peaks[name]["type"] == "lorentzian":
                params = p1[0:3]
                params = numpy.append(params, p1[-1])
                print("params: " + repr(params))
                result = ScanFitter._lorentzian(xcol, params)

            ydata[name] = result

        ScanPlot.plotCurves(xcol, ydata)

    @staticmethod
    def multiVoigtFit(scan, detector, roi, peaks):

        result = ScanAnalyzer.extract_rois(scan, {"fit_roi": roi})

        roi_intensity = numpy.array(result[detector]["fit_roi"])

        xcol = scan.scan_motors[0]
        xcol_name = scan.scan_motor_names[0]

        params = []

        bounds_low = []
        bounds_high = []

        '''
            0 : position
            1 : gauss_intensity
            2 : gauss_width
            3 : lorentzian_intensity
            4 : lorentzian_width
        '''

        peak_list = []

        for key in peaks:

            if peaks[key]["type"] == "voigt":
                params += peaks[key]["params"][0:5]

                peak_list.append(key)

                bounds_low += [10, 0, 0, 0, 0]
                bounds_high += [30, 1e10, 20, 1e10, 20]

        print(params)
        params += [peaks[key]["params"][5]]
        bounds_low += [0]
        bounds_high += [1e5]

        print(params)

        # fitfunc = lambda p, x: ScanFitter._gauss(x, p) # Target function
        # errfunc = lambda p, x, y: fitfunc(p, x) - y # Distance to the target function

        fitfunc = lambda p, x: ScanFitter._multi_voigt(x, p)  # Target function
        # errfunc = lambda p, x, y: numpy.abs(numpy.log10(fitfunc(p, x)+.1) - numpy.log10(y+.1)) # Distance to the target function

        # errfunc = lambda p: numpy.sum(numpy.abs(numpy.log10(fitfunc(p, xcol)+.1) - numpy.log10(roi_intensity+.1)))
        # errfunc = lambda p: ScanFitter.diff_func( fitfunc(p, xcol)+.1, roi_intensity+.1)

        # p1, success = optimize.least_squares(errfunc, params[:], args=(xcol, roi_intensity), bounds=(bounds_low,bounds_high))
        # p1, success = optimize.least_squares(errfunc, params[:], args=(xcol, roi_intensity))
        # p1, success = optimize.leastsq(errfunc, params[:], args=(xcol, roi_intensity))

        # print (errfunc(params[:]))

        # print("errorfunc:" +  str(errfunc(params[:])))

        # p1, success = optimize.least_squares(errfunc, params[:], args=(xcol, roi_intensity))
        print(len(params))

        try:
            # p1, success = optimize.least_squares(errfunc, params[:])

            param_tuple = tuple(params)
            print("param_tuple: " + str(param_tuple))

            # p1, success = optimize.curve_fit(ScanFitter._multi_voigt, xcol, roi_intensity, p0 = param_tuple)

            fit_fun2 = lambda xcol, *p: numpy.log10(ScanFitter._multi_voigt(xcol, *p) + 1)

            # print ("fit_fun2 : " + repr(fit_fun2(xcol, *param_tuple) ) )

            # p2, p2cov = optimize.curve_fit(fit_fun2, xcol, numpy.log10(roi_intensity+1), p0 = param_tuple, bounds=(bounds_low,bounds_high), method='dogbox')
            p2, p2cov = optimize.curve_fit(fit_fun2, xcol, numpy.log10(roi_intensity + .1), p0=param_tuple)

            print("p2cov: " + str(p2cov))

        except:
            etype, value, etraceback = sys.exc_info()
            print(etype)
            print(value)
            print("traceback: " + str(etraceback))

            traceback.print_exc()
            return

        # print ("p1 : " + repr(p1) )

        # ydata = {'data' : roi_intensity, 'before_fit' : ScanFitter._multi_voigt(xcol, params), 'after_fit' : ScanFitter._multi_voigt(xcol, p1)  }
        # ydata = {'data' : roi_intensity, 'before_fit' : ScanFitter._multi_voigt(xcol, tuple(params) ), 'after_fit' : ScanFitter._multi_voigt(xcol, tuple(p1))  }
        # ydata = {'data' : roi_intensity, 'before_fit' : ScanFitter._multi_voigt(xcol, *param_tuple ), 'curve_fit' : ScanFitter._multi_voigt(xcol, *p1), 'log_fit' : ScanFitter._multi_voigt(xcol, *p2)  }
        ydata = {'data': roi_intensity, 'before_fit': ScanFitter._multi_voigt(xcol, *param_tuple),
                 'log_fit': ScanFitter._multi_voigt(xcol, *p2)}
        # ydata = {'data' : roi_intensity, 'before_fit' : ScanFitter._multi_gauss(xcol, params) }

        # ydata = {'data' : roi_intensity, 'fit' : ScanFitter._multi_voigt(xcol, p1)  }

        for idx in range(len(peak_list)):
            name = peak_list[idx]

            print("% d: %s" % (idx, name))

            voigt_params = p2[idx * 5:idx * 5 + 5]

            voigt_params = numpy.append(voigt_params, p2[-1])

            print("voigt_params: " + repr(voigt_params) + "type: " + str(type(voigt_params)))

            result = ScanFitter._pseudo_voigt(xcol, voigt_params)

            ydata[name] = result

        ScanPlot.plotCurves(xcol, ydata)

    @staticmethod
    def diff_func(array1, array2):

        # print(" %6.4f %6.4f " % (sum(numpy.log10(array1)),sum(numpy.log10(array2))) )

        diff = numpy.sum(numpy.abs(numpy.log10(array1) - numpy.log10(array2)))

        # print (diff)

        return diff

    @staticmethod
    def multiFit(scan, detector, roi, peaks):

        result = ScanAnalyzer.extract_rois(scan, {"fit_roi": roi})

        roi_intensity = numpy.array(result[detector]["fit_roi"])

        xcol = scan.scan_motors[0]
        xcol_name = scan.scan_motor_names[0]

        params = []

        peak_list = []

        for key in peaks:

            if peaks[key]["type"] == "gauss":
                params += peaks[key]["params"][0:3]

                peak_list.append(key)

        print(params)
        params += [peaks[key]["params"][3]]

        print(params)

        fitfunc = lambda p, x: ScanFitter._multi_gauss(x, p)  # Target function
        errfunc = lambda p, x, y: numpy.sum(
            numpy.abs(numpy.log10(fitfunc(p, x)) - numpy.log10(y)))  # Distance to the target function

        # p1, success = optimize.leastsq(errfunc, params[:], args=(xcol, roi_intensity))

        p1, success = optimize.curve_fit(fitfunc, xcol, roi_intensity)

        print("p1 : " + repr(p1))

        print(fitfunc(params, xcol))

        ydata = {'data': roi_intensity, 'fit': ScanFitter._multi_gauss(xcol, p1)}

        for idx in range(len(peak_list)):
            name = peak_list[idx]

            print("% d: %s" % (idx, name))

            gauss_params = p1[idx * 3:idx * 3 + 3]

            gauss_params = numpy.append(gauss_params, p1[-1])

            print("gauss_params: " + repr(gauss_params))

            result = ScanFitter._gauss(xcol, gauss_params)

            ydata[name] = result

        ScanPlot.plotCurves(xcol, ydata)

    @staticmethod
    def _multi_gauss(x, params, nr_peaks=None):

        nr_peaks = int(len(params) / 3)

        result = numpy.zeros([len(x), ])

        for idx in range(nr_peaks):
            gauss_params = params[idx * 3:idx * 3 + 3]

            # print("gauss_params: " + repr(gauss_params))

            # gauss_params.append(0.)
            gauss_params = numpy.append(gauss_params, 0.0)
            result += ScanFitter._gauss(x, gauss_params)

        result += params[-1]

        return result

    @staticmethod
    def _multi_voigt(x, *params):

        nr_peaks = int(len(params) / 5)

        # print ("peaks: %d" % nr_peaks)
        # print ("params: " + str(params))

        result = numpy.zeros([len(x), ])

        for idx in range(nr_peaks):
            voigt_params = params[idx * 5:idx * 5 + 5]

            # print("gauss_params: " + repr(gauss_params))

            # gauss_params.append(0.)
            voigt_params = numpy.append(voigt_params, 0.0)
            result += ScanFitter._pseudo_voigt(x, voigt_params)

        result += params[-1]

        # print("multi voigt done")

        return result

    @staticmethod
    def _gauss_lorentzian(x, params):

        if not len(params) == 7:
            raise Exception("ERROR on _gauss_lorentzian: len(params) == 7 required (%d given) " % len(params))

        result = numpy.zeros([len(x), ])

        lorentzian_params = params[0:3]
        lorentzian_params = numpy.append(lorentzian_params, 0.0)

        gauss_params = params[3:6]
        gauss_params = numpy.append(gauss_params, 0.0)

        result += ScanFitter._lorentzian(x, lorentzian_params)
        result += ScanFitter._gauss(x, gauss_params)

        result += params[-1]

        return result

    @staticmethod
    def _gauss(x, params):
        '''
        copy of the MATLAB gauss function:

        function [ int ] = gauss( x, params )
        %GAUSS function
        %   x:      x-values
        %   params: 1 - sigma
        %           2 - shift
        %           3 - intensity
        %           4 - background

        gauss = @(x,params) params(3).*1./sqrt(2*pi*params(1).^2) * exp( -0.5.*((x-params(2))/params(1)).^2 ) + params(4);
        int = gauss(x,params);
        '''

        intensity = params[2] / (numpy.sqrt(2 * numpy.pi * params[0] ** 2)) * numpy.exp(
            -0.5 * ((numpy.array(x) - params[1]) / params[0]) ** 2) + params[3]

        return intensity

    @staticmethod
    def _lorentzian(x, params):
        '''

        lorentzian:

        x:  x-values
        params 0: omega_0   (maximum position)
        params 1: gamma     (width)
        params 2: intensity
        params 4: background


        f(omega) = 1/( (omega**2 - omega_0**2 + gamma**2*omega_0**2) )

        '''

        intensity = params[2] * 1 / ((x ** 2 - params[0] ** 2) ** 2 + params[1] ** 2 * params[0] ** 2) + params[3]

        return intensity

    @staticmethod
    def _pseudo_voigt(x, params):

        '''
        params:
            0 : position
            1 : gauss_intensity
            2 : gauss_width
            3 : lorentzian_intensity
            4 : lorentzian_width
            5 : background
        '''

        gauss_params = [params[2], params[0], params[1], 0]
        lorentzian_params = [params[0], params[4], params[3], 0]

        return ScanFitter._gauss(x, gauss_params) + ScanFitter._lorentzian(x, lorentzian_params) + params[5]


class ScanPlot(object):

    def __init__(self):
        pass

    @staticmethod
    def plotRois(scan, detector, rois, xcolName=None):
        '''
        uses matplotlib to plot roi intensities of scan for given detector

        
        '''
        xcol_idx = 0

        if not xcolName is None:
            for idx in range(len(scan.scan_motor_names)):
                if scan.scan_motor_names[idx] == xcolName:
                    xcol_idx = idx
                    break

        xcol = scan.scan_motors[xcol_idx]

        xcol_name = scan.scan_motor_names[xcol_idx]

        result = ScanAnalyzer.extract_rois(scan, rois)

        roi_intensity = result[detector]

        nr_plots = len(roi_intensity)

        img_size = scan.image_data[detector].shape[0]

        if "background" in roi_intensity:

            background_area = (rois["background"][2] - rois["background"][0]) * (
                    rois["background"][3] - rois["background"][1])

            print(background_area)

            background = numpy.array(roi_intensity["background"])
            nr_plots -= 1
        else:
            background = 0
            background_area = 1

        sqrt_nr = int(numpy.sqrt(nr_plots) + 0.5)
        cnt = 1
        for key in sorted(roi_intensity):

            if not key == "background":
                print(key)
                plt.subplot(sqrt_nr, sqrt_nr, cnt)

                if not key == "all":
                    roi_area = (rois[key][2] - rois[key][0]) * (rois[key][3] - rois[key][1])
                    print(roi_area)
                else:
                    roi_area = numpy.prod(img_size)

                # plt.semilogy(xcol, roi_intensity[key] )
                plt.semilogy(xcol, numpy.array(roi_intensity[key]) / roi_area - background / background_area)
                plt.ylabel(key)
                plt.xlabel(xcol_name)

                cnt += 1

        plt.show()

    @staticmethod
    def plotCurves(xcol, ydata):

        # print (ydata)

        fig, ax = plt.subplots()

        for key in ydata:
            if key == "substrate":
                ax.semilogy(xcol, numpy.array(ydata[key]), label=key, linestyle=':')
            elif key == "film":
                ax.semilogy(xcol, numpy.array(ydata[key]), label=key, linestyle=':')
            else:
                ax.semilogy(xcol, numpy.array(ydata[key]), label=key, linestyle='-')

        legend = ax.legend(loc='upper left')

        plt.show()


if __name__ == '__main__':

    # filename = '/asap3/petra3/gpfs/p08/2019/data/11006183/raw/cfosto190311_00183.fio'
    # filename = '/asap3/petra3/gpfs/p08/2019/data/11006183/raw/cfosto190311_00187.fio'
    # filename = '/asap3/petra3/gpfs/p08/2019/data/11006183/raw/nfosto190211_00297.fio'
    # rois = {"roi1" : [754, 249, 775, 265], "roi2" : [754, 249, 1500, 265], "roi3" : [800, 249, 1500, 265] }

    filename = '/asap3/petra3/gpfs/p08/2019/commissioning/c20190208_000_start19/raw/ptysz02032019_00423.fio'

    rois = {"roi1": [163, 71, 174, 76], "roi2": [164, 69, 174, 79], "roi3": [0, 0, 487, 194]}
    scan1 = Scan()

    # scan1.load_scan(filename)
    # print (scan1)

    # scan1.remove_double()
    # ScanPlot.plotRois(scan1, "p100k", rois)

    # filenames = ['/asap3/petra3/gpfs/p08/2019/data/11006183/raw/nfomao190322_01039.fio', '/asap3/petra3/gpfs/p08/2019/data/11006183/raw/nfomao190322_01040.fio']
    # rois = {"roi1" : [765, 279, 830, 290], "roi2" : [802, 279, 1500, 290], "roi3" : [820, 279, 1500, 290], "background" : [1000, 279, 1340, 290] }

    filenames = ['/asap3/petra3/gpfs/p08/2019/data/11006183/raw/nfosto190211_00286.fio',
                 '/asap3/petra3/gpfs/p08/2019/data/11006183/raw/nfosto190211_00287.fio']
    rois = {"roi1": [761, 279, 812, 290], "roi2": [761, 279, 1220, 290], "roi3": [940, 279, 1220, 290],
            "background": [1220, 279, 1340, 290]}

    scans = []
    for name in filenames:
        scan = Scan()
        scan.load_scan(name)
        scans.append(scan)
    combined_scan = ScanAnalyzer.combine_scan_image_regions(scans, [[0, 0, 880, 516], [880, 0, 1340, 516]])

    ScanAnalyzer.writeHDF("test.h5", combined_scan)

    print(combined_scan)

    try:
        print("daten: " + combined_scan.data)
    except:
        pass
    try:
        print("filename: " + combined_scan.filename)
    except:
        pass

    # ScanPlot.plotRois(combined_scan, "lambda", rois)

    # ScanFitter.multiFit(combined_scan, "lambda", rois["roi2"], {"film" : {"type" : "gauss", "params" : [0.015,0.95,2e4,4e4]}, "bck" : {"type" : "gauss", "params" : [0.15,0.95,1e3,4e4]}, "substrate" :  {"type" : "gauss", "params" : [0.003,1.01,2e5,1.2e5]} })

    ScanFitter.gaussLorentzianFit(combined_scan, "lambda", rois["roi2"],
                                  {"film": {"type": "gauss", "params": [0.015, 0.95, 2e4, 4e4]},
                                   "substrate": {"type": "lorentzian", "params": [1.01, 0.003, 1e3, 1.2e5]}})

    # filename = '/asap3/petra3/gpfs/p08/2019/data/11005886/raw/M71801_FG966_probeC_00183.fio'
    # rois = {"roi1" : [163, 71, 174, 76], "roi2" : [164, 69, 174, 79], "roi3" : [0, 0, 487, 194] }
    # scan1 = Scan()
    # scan1.load_scan(filename)
    # print(scan1)
