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
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
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


def cluster_results(reduced_data, preds, centers):
    '''
    Visualizes the PCA-reduced cluster data in two dimensions
    Adds cues for cluster centers and student-selected sample data
    :param reduced_data: pandas dataframe. the dataset transformed and cleaned
    :param preds: numpy array. teh cluster classification of each datapoint
    :param centers: numpy array. the center of the clusters
    :param pca_samples: numpy array. the sample choosen
    '''

    predictions = pd.DataFrame(preds, columns=['Cluster'])
    plot_data = pd.concat([predictions, reduced_data], axis=1)

    # Generate the cluster plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Color map
    cmap = sns.color_palette('cubehelix', 12)

    # Color the points based on assigned cluster
    for i, cluster in plot_data.groupby('Cluster'):
        cluster.plot(ax=ax, kind='scatter', x='Dimension 1', y='Dimension 2',
                     color=cmap[i],
                     label='Cluster %i' % (i),
                     s=30)

    # Plot centers with indicators
    for i, c in enumerate(centers):
        ax.scatter(x=c[0], y=c[1], color='white', edgecolors='black',
                   alpha=1, linewidth=2, marker='o', s=200)
        ax.scatter(x=c[0], y=c[1], marker='$%d$' % (i), alpha=1, s=100)

    # Set plot title
    s_title = 'Cluster Learning on PCA-Reduced Data - Centroids Marked by'
    s_title += ' Number\n'
    ax.set_title(s_title, fontsize=16)


def pca_results(good_data, pca):
    '''
    Create a DataFrame of the PCA results. Includes dimension feature weights
    and explained variance Visualizes the PCA results
    :param good_data: DataFrame. all dataset log transformed with 6 columns
    :param pca: Sklearn Object. a PCA decomposition object already fitted
    '''
    # Dimension indexing
    dimensions = dimensions = ['Dimension {}'.format(i)
                               for i in range(1, len(pca.components_)+1)]

    # PCA components
    components = pd.DataFrame(np.round(pca.components_, 4),
                              columns=good_data.keys())
    components.index = dimensions

    # PCA explained variance
    ratios = pca.explained_variance_ratio_.reshape(len(pca.components_), 1)
    variance_ratios = pd.DataFrame(np.round(ratios, 4),
                                   columns=['Explained Variance'])
    variance_ratios.index = dimensions

    # reshape the data to be plotted
    df_aux = components.unstack().reset_index()
    df_aux.columns = ['Feature', 'Dimension', 'Variance']

    # Create a bar plot visualization
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the feature weights as a function of the components
    sns.barplot(x='Dimension', y='Variance', hue='Feature', data=df_aux, ax=ax)
    ax.set_ylabel('Feature Weights')
    ax.set_xlabel('')
    ax.set_xticklabels(dimensions, rotation=0)

    # Display the explained variance ratios
    for i, ev in enumerate(pca.explained_variance_ratio_):
        ax.text(i-0.40, ax.get_ylim()[1] + 0.05,
                'Explained Variance\n          %.4f' % (ev))

    # insert a title
    # ax.set_title('PCA Explained Variance Ratio',
    #              fontsize=16, y=1.10)

    # Return a concatenated DataFrame
    return pd.concat([variance_ratios, components], axis=1)


def simple_counts(s_fname, s_agent):
    '''
    Analyze thew log files generated by the agents
    :param s_fname: string. Name of the log file
    :param s_agent: string. Name of the agent in the logfile
    '''
    with open(s_fname) as fr:
        d_cumrewr = defaultdict(lambda: defaultdict(float))
        d_position = defaultdict(lambda: defaultdict(int))
        d_reward = defaultdict(int)
        f_reward = 0.
        f_count_step = 0
        last_reward = 0.
        i_trial = 0

        for idx, row in enumerate(fr):
            if row == '\n':
                continue
            s_aux = row.strip().split(';')[1]
            # extract desired information
            if '{}.update'.format(s_agent) in s_aux:
                s_x = row.split('time = ')[1].split(' ')[1].split(',')[0]
                s_date_all = s_x
                s_x = row.split('time = ')[1].split(' ')[1].split(',')[0][:-3]
                s_date = s_x
                ts_date_all = pd.to_datetime(s_date_all)
                ts_date = pd.to_datetime(s_date + ':00')
                last_reward = float(s_aux.split('reward = ')[1].split(',')[0])
                f_x = float(s_aux.split('position = ')[1].split(',')[0])
                d_position[i_trial+1][ts_date_all] = f_x
                d_cumrewr[i_trial+1][ts_date] = f_reward + last_reward
                f_reward += last_reward
                f_count_step += 1.
            # store cumulative data
            elif 'Environment.reset' in s_aux:
                if f_count_step > 0:
                    d_reward[i_trial+1] = f_reward / f_count_step
                i_trial += 1
                b_already_finish = False
                f_count_step = 0
                f_reward = 0

        return d_cumrewr, d_reward, d_position
