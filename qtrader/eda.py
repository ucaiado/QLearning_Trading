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
import matplotlib.dates as dates
from matplotlib import ticker as mticker
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


def make_df(d_data):
    '''
    Reshape the data passed to acumulate the pnl from previous days
    :param d_data: dict. PnL data from tests performed
    '''
    df_aux = pd.DataFrame(d_data)
    df_filter = pd.Series([x.day for x in df_aux.index])
    df_aux2 = pd.DataFrame(np.zeros(df_aux.shape))
    df_aux2.index = df_aux.index
    df_aux2.columns = df_aux.columns
    df_aux3 = df_aux.shift()[(df_filter != df_filter.shift()).values]
    df_aux3 = df_aux3.fillna(0.)
    df_aux2.ix[list(df_aux3.index)] += df_aux3
    df_aux2 = df_aux2.cumsum()
    return df_aux + df_aux2

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
    Visualizes the reduced cluster data in two dimensions
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
    # cmap = sns.color_palette('cubehelix', 12)
    cmap = sns.color_palette('Set2', 12)

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
    s_title = 'Cluster Learning on Reduced Data - Centroids Marked by'
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
        d_cumrewr = {'test': defaultdict(lambda: defaultdict(float)),
                     'train': defaultdict(lambda: defaultdict(float))}
        d_pnl = {'test': defaultdict(lambda: defaultdict(float)),
                 'train': defaultdict(lambda: defaultdict(float))}
        # d_position = {'test': defaultdict(lambda: defaultdict(int)),
        #               'train': defaultdict(lambda: defaultdict(int))}
        d_cumrewr = {'test': defaultdict(lambda: defaultdict(float)),
                     'train': defaultdict(lambda: defaultdict(float))}
        d_reward = {'test': defaultdict(int),
                    'train': defaultdict(int)}
        d_delta_pnl = defaultdict(int)
        d_action = defaultdict(int)
        f_reward = 0.
        f_count_step = 0
        last_reward = 0.
        i_trial = 0
        s_phase = 'train'

        for idx, row in enumerate(fr):
            if row == '\n':
                continue
            s_aux = row.strip().split(';')[1]
            # extract desired information
            if '{}.update'.format(s_agent) in s_aux:
                # s_x = row.split('time = ')[1].split(' ')[1].split(',')[0]
                s_x = row.split('time = ')[1].split(',')[0]
                s_date_all = s_x
                s_x = s_date_all[:-3]
                s_date = s_x
                ts_date_all = pd.to_datetime(s_date_all)
                ts_date = pd.to_datetime(s_date + ':00')
                last_reward = float(s_aux.split('reward = ')[1].split(',')[0])
                f_x = float(s_aux.split('position = ')[1].split(',')[0])
                # d_position[s_phase][i_trial+1][ts_date_all] = f_x
                d_cumrewr[s_phase][i_trial+1][ts_date] = f_reward + last_reward
                f_reward += last_reward
                f_count_step += 1.
                if 'delta_pnl = ' in s_aux:
                    f_val = float(s_aux.split('delta_pnl = ')[1].split(',')[0])
                    d_delta_pnl[int(f_val)] += 1
                if ', action = ' in s_aux:
                    s_action = s_aux.split(', action = ')[1].split(',')[0]
                    d_action[s_action] += 1
                if ', pnl = ' in s_aux:
                    s_action = s_aux.split(', pnl = ')[1].split(',')[0]
                    d_pnl[s_phase][i_trial+1][ts_date] = float(s_action)
            elif 'Trial Ended' in s_aux:
                # store cumulative data
                if f_count_step > 0:
                    d_reward[s_phase][i_trial+1] = f_reward / f_count_step
                i_trial += 1
                b_already_finish = False
                f_count_step = 0
                f_reward = 0
            elif 'run(): Starting testing phase !' in s_aux:
                i_trial = 0
                s_phase = 'test'

        d_summary = {}
        d_summary['cumulative_reward'] = d_cumrewr
        d_summary['avg_reward'] = d_reward
        # d_summary['position'] = d_position
        d_summary['delta_pnl'] = d_delta_pnl
        d_summary['pnl'] = d_pnl
        d_summary['action'] = d_action

        return d_summary


def count_by_k_gamma(s_fname, s_agent, s_split):
    '''
    Analyze thew log files generated by the agents, separating the information
    by k or gamma values
    :param s_fname: string. Name of the log file
    :param s_agent: string. Name of the agent in the logfile
    :param s_split: string. 'gamma' or 'k'. Key to use to split data
    '''
    assert s_split in ['k', 'gamma'], 's_split should be k or gamma'
    with open(s_fname) as fr:
        d_rtn = {}
        d_gamma = {}
        d_delta_pnl = defaultdict(int)
        d_action = defaultdict(int)
        f_reward = 0.
        f_count_step = 0
        last_reward = 0.
        i_trial = 0

        for idx, row in enumerate(fr):
            if row == '\n':
                continue
            s_aux = row.strip().split(';')[1]
            # extract desired information
            if '.choose_an_action()' in s_aux:
                if s_split == 'gamma':
                    s_key = s_aux.split('gamma = ')[1].split(',')[0]
                if s_split == 'k':
                    s_key = s_aux.split('k = ')[1].split(',')[0]
                if s_key not in d_rtn:
                    d_rtn[s_key] = defaultdict(lambda: defaultdict(float))
                    i_trial = 0
                    b_already_finish = False
                    f_count_step = 0
                    f_reward = 0
            if '{}.update'.format(s_agent) in s_aux:
                s_x = row.split('time = ')[1].split(',')[0]
                s_date_all = s_x
                s_x = s_date_all[:-3]
                s_date = s_x
                ts_date_all = pd.to_datetime(s_date_all)
                ts_date = pd.to_datetime(s_date + ':00')
                last_reward = float(s_aux.split('reward = ')[1].split(',')[0])
                f_x = float(s_aux.split('position = ')[1].split(',')[0])
                f_reward += last_reward
                f_count_step += 1.
                if ', pnl = ' in s_aux:
                    s_action = s_aux.split(', pnl = ')[1].split(',')[0]
                    f_aux = float(s_action)
                    d_rtn[s_key][i_trial+1][ts_date] = f_aux
            elif 'Trial Ended' in s_aux:
                i_trial += 1
                b_already_finish = False
                f_count_step = 0
                f_reward = 0

        return d_rtn


def plot_train_test_sim(d_rtn):
    '''
    Plot the PnL curves from simulations to compare the performance of each
    policy learned on the traning phase in on the test phase
    :param d_rtn: dict. Data from simulation
    '''
    f, na_ax = plt.subplots(2, 5, sharex=True, sharey=True)
    df_test = make_df(d_rtn['pnl']['test'])
    df_train = make_df(d_rtn['pnl']['train'])
    for ax1, idx in zip(na_ax.ravel(), range(df_test.shape[1])):
        df_test.iloc[:, idx].reset_index(drop=True).plot(legend=True,
                                                         label='Test',
                                                         ax=ax1)
        df_train.iloc[:, idx].reset_index(drop=True).plot(legend=True,
                                                          label='Train',
                                                          ax=ax1)
        ax1.set_title('fold: {}'.format(idx+1), fontsize=10)
        ax1.xaxis.set_ticklabels([])
        ax1.set_ylabel('PnL', fontsize=8)
        ax1.set_xlabel('Time Step', fontsize=8)
    f.tight_layout()
    s_title = 'Cumulative PnL from LearningAgent_k\n'
    f.suptitle(s_title, fontsize=16, y=1.03)


def plot_cents_changed(archive, archive2):
    '''
    Plot price changes from some of the files passed
    :param archive: Zipfile object. files holder from PETR4
    :param archive2: Zipfile object. files holder from BOVA11
    '''
    l_fnames = archive.infolist()
    l_fnames2 = archive2.infolist()
    # load data
    df_prices = pd.read_csv(archive.open(l_fnames[6]),
                            index_col=0,
                            parse_dates=['Date'])
    df_prices = df_prices[df_prices.Type == 'TRADE']
    for idx in [16, 26, 36]:
        df_aux = pd.read_csv(archive.open(l_fnames[idx]),
                             index_col=0,
                             parse_dates=['Date'])
        df_aux = df_aux[df_aux.Type == 'TRADE']
        df_prices = pd.concat([df_prices, df_aux], ignore_index=True)
    df_prices.index = df_prices.Date
    quotes = df_prices.Price.resample('5min').agg({'OPEN': 'first',
                                                   'HIGH': 'max',
                                                   'LOW': 'min',
                                                   'CLOSE': 'last'})
    quotes.dropna(inplace=True)
    # load data from BOVA11
    df_prices2 = pd.read_csv(archive2.open(l_fnames2[0]),
                             sep='\t',
                             index_col=0,
                             parse_dates=['DATE'],
                             dayfirst=True,
                             decimal=',')
    quotes2 = df_prices2.PRICE.resample('5min').agg({'OPEN': 'first',
                                                     'HIGH': 'max',
                                                     'LOW': 'min',
                                                     'CLOSE': 'last'})
    quotes2.dropna(inplace=True)
    # filer data
    na_hour = np.array([x.hour*60 + x.minute for x in quotes.index])
    quotes = quotes[(na_hour >= (10 * 60 + 30)) & (na_hour <= (16 * 60 + 30))]
    na_days = np.array([x.day for x in quotes.index])
    quotes2 = quotes2.ix[quotes.index, :]
    quotes2.fillna(method='ffill', inplace=True)
    # plot price changes in cents
    fig, na_ax = plt.subplots(1, 4, sharey=True, figsize=(11, 6))
    na_ax = na_ax.ravel()
    na_unique = np.unique(na_days)
    l_idx = [6, 16, 26, 36]
    xfmt = dates.DateFormatter('%H:%M')
    l_last = []
    for idx, ax in enumerate(na_ax):
        df_plot = quotes[na_days == na_unique[idx]]

        df_plot2 = quotes2[na_days == na_unique[idx]]
        f_convert = df_plot.iloc[0].CLOSE / df_plot2.iloc[0].CLOSE
        df_plot2 = (df_plot2 * f_convert).round(2)

        df_plot.index.name = None
        df_plot = (df_plot-df_plot.shift()).cumsum()

        df_plot2.index.name = None
        df_plot2 = (df_plot2-df_plot2.shift()).cumsum()

        l_last.append({'PETR4': df_plot.iloc[-1].CLOSE,
                       'BOVA11': df_plot2.iloc[-1].CLOSE})

        (df_plot.CLOSE * 100).plot(ax=ax, label='PETR4', legend=True)
        (df_plot2.CLOSE * 100).plot(ax=ax, label='BOVA11', legend=True)
        if idx in [0]:
            ax.set_ylabel('cents', fontsize=12)
        if idx in [0, 1, 2, 3]:
            ax.set_xlabel('Time', fontsize=12)
        ax.yaxis.set_major_locator(mticker.MaxNLocator(7))
        ax.xaxis.set_major_locator(mticker.MaxNLocator(3))
        ax.grid(axis='x')
        ax.set_title('idx: {}'.format(l_idx[idx]), fontsize=12)

    fig.tight_layout()
    return pd.DataFrame(l_last)
