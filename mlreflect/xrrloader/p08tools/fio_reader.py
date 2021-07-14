# -*- coding: utf-8 -*-
"""
Created on Sun Sep 23 17:18:21 2018

@author: Florian
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re


def read(filename, header_only=False):
    motor_positions = {}

    data_block = False
    param_block = False
    comment_block = False

    data_columns = False

    column_names = []

    scan_cmd = None

    header_info = {}

    rois = {}

    data = {}

    file = open(filename, 'r')

    for line in file:
        # print( line )

        if line.find('%c') > -1:
            data_block = False
            param_block = False
            comment_block = True
            # print('entering comment block')
            continue
        elif line.find('%p') > -1:
            data_block = False
            param_block = True
            comment_block = False
            # print('entering parameter block')
            continue
        elif line.find('%d') > -1:
            data_block = True
            param_block = False
            comment_block = False
            # print('entering data block')
            continue
        elif line.find('!') > -1:
            continue

        if param_block:
            if line.find('=', 1) > -1:
                try:
                    spl = line.strip().split('=')
                    # print("%s = %f" % (spl[0], float(spl[1])))
                    motor_positions[spl[0].strip()] = float(spl[1])
                except:
                    try:
                        spl = line.strip().split('=')

                        if spl[0].strip().lower() == 'ubmatrix':
                            line = line.replace(';', ',')
                            spl = line.strip().split('=')

                            # print (spl[1])
                            bra_open = []
                            for idx in range(len(spl[1])):
                                if spl[1][idx] == '[':
                                    bra_open.append(idx)

                            bra_close = []
                            for idx in range(len(spl[1])):
                                if spl[1][idx] == ']':
                                    bra_close.append(idx)

                            first_row = re.sub(' +', ' ', spl[1][bra_open[1] + 1: bra_close[0]].strip())
                            first_row = [float(nr) for nr in first_row.split()]

                            sec_row = re.sub(' +', ' ', spl[1][bra_open[2] + 1: bra_close[1]]).strip()
                            sec_row = [float(nr) for nr in sec_row.split()]

                            thir_row = re.sub(' +', ' ', spl[1][bra_open[3] + 1: bra_close[2]].strip())
                            thir_row = [float(nr) for nr in thir_row.split()]

                            matrix = [first_row, sec_row, thir_row]

                            header_info['ubmatrix'] = matrix
                        elif spl[0].strip().lower() == 'signalcounter':
                            header_info['signalcounter'] = spl[1].strip()
                        elif not spl[0].strip().find('roi') == -1:
                            thisrois = [float(nr) for nr in spl[1].strip()[1:-1].split(',')]

                            roi_name = spl[0].split(' ')[1].strip()

                            cur_rois = {}

                            cnt = 1
                            for idx in range(0, len(thisrois), 4):
                                cur_roi = thisrois[idx:idx + 4]
                                cur_rois["roi%d" % cnt] = cur_roi
                                cnt += 1

                            rois[roi_name] = cur_rois


                    except:
                        print('error: %s' % line)
        elif comment_block:
            if line.find('scan') > -1 or line.find('mesh') > -1:
                scan_cmd = line.strip()
        elif data_block and not data_columns:
            if line.find('Col ') > -1:
                spl = line.split()
                column_names.append(spl[2])

            elif len(column_names) > 0:

                if header_only:
                    break

                spl = line.split()

                for idx in range(len(column_names)):
                    try:
                        data[column_names[idx]] = [float(spl[idx])]
                    except:
                        data[column_names[idx]] = [float('nan')]

                data_columns = True

        elif data_columns:

            spl = line.split()

            for idx in range(len(column_names)):
                try:
                    data[column_names[idx]].append(float(spl[idx]))
                except:
                    data[column_names[idx]].append(float('nan'))

        # print ( line.find('=') )

    file.close()

    header_info["scan_cmd"] = scan_cmd
    header_info["rois"] = rois

    return motor_positions, column_names, data, header_info


if __name__ == '__main__':

    import cProfile

    cProfile.run("read('./data/test_00065.fio')", sort="tottime")

    header, column_names, data = read('./data/test_00065.fio')

    # for key in sorted(header):
    #    print ("%s = %f" % (key, header[key]))

    for col in column_names:
        print(col)

    if 'om' in column_names:
        print(data['om'])
        print(len(data['om']))
