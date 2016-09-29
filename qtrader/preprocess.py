#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Preprocess the dataset to make it easy to use in simulation

@author: ucaiado

Created on 09/01/2016
"""

import zipfile
import csv
import numpy as np
import pandas as pd
import pickle
import time


def make_zip_file(s_fname):
    '''
    Process a zip file and convert in another one with files more easly
    translate to the order book
    :param s_fname: string. zip file path
    '''
    f_start = time.time()
    archive = zipfile.ZipFile(s_fname, 'r')

    f_total = 0.
    f_bid = 0.
    f_ask = 0.
    for i, x in enumerate(archive.infolist()):
        s_fname = 'data/petr4_0725_0818_2/' + x.filename
        l_hold = []
        with open(s_fname, 'w') as fw:
            for idx, d_row in enumerate(csv.DictReader(archive.open(x))):
                # check if should read row
                if int(d_row['Size']) % 100 != 0:
                    continue
                if float(d_row['Price']) == 0:
                    continue
                # check if it is a trade
                if d_row['Type'] == 'TRADE' and idx > 0:
                    l_hold.append(d_row.copy())
                    continue
                if len(l_hold) == 1:
                    d_aux = l_hold[0]
                    l_aux = [d_aux[''], d_aux['Date'], d_aux['Type'],
                             d_aux['Price'], d_aux['Size']]
                    fw.write(','.join(l_aux) + '\n')
                    l_hold = []
                elif len(l_hold) > 1:
                    # hold info to use in msgs
                    i_id = int(l_hold[0][''])
                    s_time = l_hold[0]['Date']
                    s_last = ''
                    i_qty = 0
                    # loop the list
                    l_print = []
                    for d_aux in l_hold:
                        if d_aux['Price'] != s_last:
                            if s_last != '':
                                # check if the price i a trade at bid or ask
                                if float(s_last) > f_ask:
                                    s_msg = '{},{},ASK,{},{}\n'.format(i_id,
                                                                       s_time,
                                                                       s_last,
                                                                       i_qty)
                                    fw.write(s_msg)
                                    i_id += 1
                                elif float(s_last) < f_bid:
                                    s_msg = '{},{},BID,{},{}\n'.format(i_id,
                                                                       s_time,
                                                                       s_last,
                                                                       i_qty)
                                    fw.write(s_msg)
                                    i_id += 1
                                # print the trade
                                for d_print in l_print:
                                    s_msg = '{},{},TRADE,{},{}\n'
                                    s_msg = s_msg.format(i_id,
                                                         s_time,
                                                         d_print['Price'],
                                                         d_print['Size'])
                                    fw.write(s_msg)
                                    i_id += 1
                                l_print = []
                            # clean parameters
                            s_last = d_aux['Price']
                            i_qty = 0

                        i_qty += int(d_aux['Size'])
                        l_print.append(d_aux.copy())
                    # print the last element
                    if s_last != '':
                        if len(l_hold) > 1:
                            if float(s_last) > f_ask:
                                if float(s_last) == float(d_row['Price']):
                                    s_msg = '{},{},ASK,{},{}\n'
                                    i_qaux = 0
                                    if d_row['Type'] == 'ASK':
                                        i_qaux = int(d_row['Size'])
                                    i_qaux += i_qty
                                    s_msg = s_msg.format(i_id,
                                                         s_time,
                                                         s_last,
                                                         i_qaux)
                                    fw.write(s_msg)
                                    i_id += 1
                                else:
                                    s_msg = '{},{},ASK,{},{}\n'.format(i_id,
                                                                       s_time,
                                                                       s_last,
                                                                       i_qty)
                                    fw.write(s_msg)
                                    i_id += 1
                            elif float(s_last) < f_bid:
                                if float(s_last) == float(d_row['Price']):
                                    i_qaux = 0
                                    if d_row['Type'] == 'BID':
                                        i_qaux = int(d_row['Size'])
                                    i_qaux += i_qty
                                    s_msg = '{},{},BID,{},{}\n'.format(i_id,
                                                                       s_time,
                                                                       s_last,
                                                                       i_qaux)
                                    fw.write(s_msg)
                                    i_id += 1
                                else:
                                    s_msg = '{},{},BID,{},{}\n'.format(i_id,
                                                                       s_time,
                                                                       s_last,
                                                                       i_qty)
                                    fw.write(s_msg)
                                    i_id += 1
                        # print the trade
                        for d_print in l_print:
                            s_msg = '{},{},TRADE,{},{}\n'
                            s_msg = s_msg.format(i_id,
                                                 s_time,
                                                 d_print['Price'],
                                                 d_print['Size'])
                            fw.write(s_msg)
                            i_id += 1
                        l_print = []
                    # clean the list
                    l_hold = []

                # print header
                if idx == 0:
                    fw.write(',Date,Type,Price,Size\n')
                # follow the best bid and ask
                if d_row['Type'] == 'BID':
                    f_bid = float(d_row['Price'])
                elif d_row['Type'] == 'ASK':
                    f_ask = float(d_row['Price'])
                # print file
                l_aux = [d_row[''], d_row['Date'], d_row['Type'],
                         d_row['Price'], d_row['Size']]
                fw.write(','.join(l_aux) + '\n')
    #     break  # remove it

    print "run in {:0.2f} seconds".format(time.time() - f_start)


class ClusterScaler(object):
    '''
    Handler of all the process to scale the input space from the learner
    '''
    def __init__(self):
        '''
        Initialize a Scaler object
        '''
        self.kmeans = pickle.load(open('data/kmeans.dat', 'r'))
        self.pca = pickle.load(open('data/pca.dat', 'r'))
        self.d_scale = {}
        self.d_scale['OFI'] = pickle.load(open('data/scale_ofi.dat', 'r'))
        self.d_scale['qBID'] = pickle.load(open('data/scale_qbid.dat', 'r'))
        scale_aux = pickle.load(open('data/scale_bookratio.dat', 'r'))
        self.d_scale['BOOK_RATIO'] = scale_aux
        self.d_scale['LOG_RET'] = pickle.load(open('data/logret.dat', 'r'))

    def transform(self, d_feat):
        '''
        Return the cluster of the input data
        :param d_feat: dictionary. Original Input data from one instamce
        '''
        # scale the features passed
        # print d_feat  # [DEBUG]
        d_data = {}
        d_data['OFI'] = d_feat['OFI']
        d_data['qBID'] = np.log(d_feat['qBID'])
        d_data['BOOK_RATIO'] = np.log(d_feat['BOOK_RATIO'])
        d_data['LOG_RET'] = d_feat['LOG_RET']
        for s_key in ['OFI', 'qBID', 'BOOK_RATIO', 'LOG_RET']:
            f_value = np.array([1. * d_data[s_key]]).reshape(1, -1)
            d_data[s_key] = float(self.d_scale[s_key].transform(f_value))
        # aplpy PCA to reduce to two dimensions
        na_val_pca = pd.Series(d_data).values.reshape(1, -1)
        na_val_pca = self.pca.transform(na_val_pca)
        # return the cluster (from 10) using kmeans
        return int(self.kmeans.predict(na_val_pca))


class LessClustersScaler(object):
    '''
    Handler of all the process to scale the input space from the learner
    '''
    def __init__(self):
        '''
        Initialize a Scaler object
        '''
        self.kmeans = pickle.load(open('data/kmeans.dat_2', 'r'))
        self.d_scale = {}
        self.d_scale['OFI'] = pickle.load(open('data/scale_ofi.dat_2', 'r'))
        scale_aux = pickle.load(open('data/scale_bookratio.dat_2', 'r'))
        self.d_scale['BOOK_RATIO'] = scale_aux

    def transform(self, d_feat):
        '''
        Return the cluster of the input data
        :param d_feat: dictionary. Original Input data from one instamce
        '''
        # scale the features passed
        # print d_feat  # [DEBUG]
        d_data = {}
        d_data['OFI'] = d_feat['OFI']
        d_data['BOOK_RATIO'] = np.log(d_feat['BOOK_RATIO'])
        for s_key in ['OFI', 'BOOK_RATIO']:
            f_value = np.array([1. * d_data[s_key]]).reshape(1, -1)
            d_data[s_key] = float(self.d_scale[s_key].transform(f_value))

        # return the cluster (from 10) using kmeans
        na_val = pd.Series(d_data).values.reshape(1, -1)
        return int(self.kmeans.predict(na_val))


class ZeroOneScaler(object):
    '''
    Handler of all the process to scale the input space from the learner
    '''
    def __init__(self):
        '''
        Initialize a Scaler object
        '''
        self.d_scale = {}
        self.d_scale['OFI'] = pickle.load(open('data/scale_ofi.dat', 'r'))
        scale_aux = pickle.load(open('data/scale_bookratio.dat', 'r'))
        self.d_scale['BOOK_RATIO'] = scale_aux

    def transform(self, d_feat):
        '''
        Return a list of the 0-1 representation of each the state
        :param d_feat: dictionary. Original Input data from one instamce
        '''
        # scale the features passed
        # print d_feat  # [DEBUG]
        d_data = {}
        d_data['OFI'] = d_feat['OFI']
        d_data['BOOK_RATIO'] = np.log(d_feat['BOOK_RATIO'])
        for s_key in ['OFI', 'BOOK_RATIO']:
            f_value = np.array([1. * d_data[s_key]]).reshape(1, -1)
            d_data[s_key] = float(self.d_scale[s_key].transform(f_value))
        # round the numbers
        l_rtn = [int(d_data['OFI'] * 10), int(d_data['BOOK_RATIO'] * 10)]
        # limit the numbers
        for idx in [0, 1]:
            if l_rtn[idx] > 10:
                l_rtn[idx] = 10
            if l_rtn[idx] < 0:
                l_rtn[idx] = 0

        # return the cluster (from 10) using kmeans
        return l_rtn
