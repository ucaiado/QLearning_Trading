#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Perform different statistical data analysis in log files produced by simulation
and create visualizations of the input space

@author: ucaiado

Created on 09/18/2016
"""
from collections import defaultdict
import csv
import numpy as np
import pandas as pd
import zipfile


'''
Begin help functions
'''


def measure_e_n(row, last_best):
    '''
    Measure the e_n of the current event
    :param row: dictionary. current row from the file
    :param last_best: tuple. best price and best quantity
    '''
    e_n = 0
    if row['Type'] == 'BID':
        e_n += (row['Price'] >= last_best[0]) * row['Size']
        e_n -= (row['Price'] <= last_best[0]) * last_best[1]
    elif row['Type'] == 'ASK':
        e_n -= (row['Price'] <= last_best[0]) * row['Size']
        e_n += (row['Price'] >= last_best[0]) * last_best[1]
    return e_n


def convert_float_to_time(f_time):
    '''
    Converst number of seconds in string time format
    :param f_time: float. number of seconds
    '''
    i_hour = int(f_time / 3600)
    i_minute = int((f_time - i_hour * 3600) / 60)
    i_seconds = int((f_time - i_hour * 3600 - i_minute * 60))
    return '{:02d}:{:02d}:{:02d}'.format(i_hour, i_minute, i_seconds)


'''
End help functions
'''


def test_ofi_indicator(s_fname, f_min_time=10.):
    '''
    Create a file with the OFI of the given files by each time bucket
    :param s_fname: string. The zip file where is the information
    :param f_min_time: float. Number of seconds to aggreagate the information
    '''
    fw_out = open('data/ofi_petr.txt', 'w')  # data output
    fw_out.write('TIME\tOFI\tDELTA_MID\tLOG_RET\tqBID\tBOOK_RATIO\n')
    archive = zipfile.ZipFile(s_fname, 'r')
    d_best_price = {'BID': (0., 0.), 'ASK': (0., 0.)}

    # read only the first file inside the ZIP file
    l_fnames = archive.filelist
    x = l_fnames[0]
    f_ofi = 0.
    f_mid = None
    f_next_time = 10 * 3600 + 5 * 60 + f_min_time
    for idx_row, row in enumerate(csv.DictReader(archive.open(x))):
        if idx_row == 0:
            f_first_price = row['Price']
        # I dont need to deal with trades
        if row['Type'] in ['BID', 'ASK']:
            # converte string para float
            row['Price'] = float(row['Price'].replace(',', '.'))
            row['Size'] = float(row['Size'])
            f_current_time = sum([float(x)*60**(2.-i_aux) for i_aux, x in
                                 enumerate(row['Date'][-8:].split(':'))])
            if f_current_time > f_next_time:
                # imprime resultado
                s_time = convert_float_to_time(f_next_time)
                f_change = 0
                f_logrtn = 0.
                if f_mid:
                    f_curent_mid = (d_best_price['ASK'][0] +
                                    d_best_price['BID'][0])/2.
                    f_change = int((f_curent_mid - f_mid)/0.01)
                    f_logrtn = np.log((f_curent_mid/f_mid))
                f_mid = (d_best_price['ASK'][0] + d_best_price['BID'][0])/2.
                s_txt = '{}\t{}\t{}\t{}\t{}\t{}\n'
                f_ratio = d_best_price['BID'][1] * 1. / d_best_price['ASK'][1]
                s_out = s_txt.format(s_time,
                                     f_ofi,
                                     f_change,
                                     f_logrtn,
                                     d_best_price['BID'][1],
                                     f_ratio)
                fw_out.write(s_out)
                # reselt counter
                f_ofi = 0
                # print info in f_min_time seconds
                f_next_time = (int(f_current_time/f_min_time) + 1)*f_min_time
            elif abs(f_current_time - f_next_time) > 3600:
                # new day
                f_next_time = 10 * 3600 + 5 * 60 + f_min_time
                f_mid = None
                f_ofi = 0
            # compare to last info
            last_best = d_best_price[row['Type']]
            f_e_n = measure_e_n(row, last_best)
            # update the last bests
            d_best_price[row['Type']] = (row['Price'], row['Size'])
            row['Date'] = row['Date'][-8:]
            f_ofi += f_e_n
