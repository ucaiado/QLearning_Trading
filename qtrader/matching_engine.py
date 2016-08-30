#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Implement different matching engines to handle the actions taken by each agent
in the environment

@author: ucaiado

Created on 08/19/2016
"""
import random
import logging
import zipfile
import csv
import book

# global variable
DEBUG = False

'''
Begin help functions
'''

'''
End help functions
'''


def translate_row(idx, row, my_book):
    '''
    Translate a line from a file of the bloomberg level I data
    :param idx: integer.
    :param row: dict.
    :param my_book: Book object.
    '''
    l_msg = []
    row['Price'] = float(row['Price'])
    row['Size'] = float(row['Size'])
    if row['Price'] == 0. or row['Size'] % 100 != 0:
        return l_msg
    if row['Type'] == 'TRADE':
        # check which side was affected
        # test bid side
        s_side = 'BID'
        obj_price = my_book.get_orders_by_price(s_side, b_rtn_obj=True)
        if obj_price:
            if row['Price'] > obj_price.f_price:
                obj_price = None
        # test ask side, if it is the case
        if not obj_price:
            s_side = 'ASK'
            obj_price = my_book.get_orders_by_price(s_side, b_rtn_obj=True)
            if obj_price:
                if row['Price'] < obj_price.f_price:
                    return l_msg
            else:
                return l_msg

        # in case the traded qty is bigger than the price level qty
        if obj_price.i_qty < row['Size']:
            return l_msg
        # translate row in message
        i_qty = row['Size']
        for idx_ord, order_aux in obj_price.order_tree.nsmallest(1000):
            # define how much should be traded
            i_qty_traded = order_aux['total_qty_order']
            i_qty_traded -= order_aux['traded_qty_order']
            i_qty_traded = min(i_qty, i_qty_traded)
            i_qty -= i_qty_traded
            # define the status of the message
            if order_aux['total_qty_order'] == i_qty_traded:
                s_status = 'Filled'
            else:
                s_status = 'Partially Filled'
            assert i_qty >= 0, 'Qty traded smaller than 0'
            # create the message
            d_rtn = {'agent_id': order_aux['agent_id'],
                     'instrumento_symbol': 'PETR4',
                     'order_id': idx_ord,
                     'order_entry_step': idx,
                     'new_order_id': idx_ord,
                     'order_price': row['Price'],
                     'order_side': s_side,
                     'order_status': s_status,
                     'total_qty_order': order_aux['total_qty_order'],
                     'traded_qty_order': i_qty_traded,
                     'agressor_indicator': 'Passive'}
            l_msg.append(d_rtn.copy())

    else:
        # recover the best price
        f_best_price = my_book.get_best_price(row['Type'])
        i_order_id = my_book.i_last_order_id + 1
        # check if there is orders in the row price
        obj_ordtree = my_book.get_orders_by_price(row['Type'],
                                                  row['Price'])
        if obj_ordtree:
            # cant present more than 2 orders (mine and market)
            assert len(obj_ordtree) <= 2, 'More than two offers'
            # get the first order
            obj_order = obj_ordtree.nsmallest(1)[0][1]
            # check if should cancel the best price
            b_cancel = False
            if row['Type'] == 'BID' and row['Price'] < f_best_price:
                # check if the price in the row in smaller
                obj_ordtree2 = my_book.get_orders_by_price(row['Type'])
                best_order = obj_ordtree2.nsmallest(1)[0][1]
                d_rtn = best_order.d_msg
                d_rtn['order_status'] = 'Canceled'
                l_msg.append(d_rtn.copy())
            elif row['Type'] == 'ASK' and row['Price'] > f_best_price:
                obj_ordtree2 = my_book.get_orders_by_price(row['Type'])
                best_order = obj_ordtree2.nsmallest(1)[0][1]
                d_rtn = best_order.d_msg
                d_rtn['order_status'] = 'Canceled'
                l_msg.append(d_rtn.copy())
            # replace the current order
            i_old_id = obj_order.main_id
            i_new_id = obj_order.main_id
            if row['Size'] > obj_order['total_qty_order']:
                i_new_id = my_book.i_last_order_id + 1
                d_rtn = obj_order.d_msg
                d_rtn['order_status'] = 'Canceled'
                l_msg.append(d_rtn.copy())
            # Replace the order
            d_rtn = {'agent_id': 10,
                     'instrumento_symbol': 'PETR4',
                     'order_id': i_old_id,
                     'order_entry_step': idx,
                     'new_order_id': i_new_id,
                     'order_price': row['Price'],
                     'order_side': row['Type'],
                     'order_status': 'Replaced',
                     'total_qty_order': row['Size'],
                     'traded_qty_order': 0,
                     'agressor_indicator': 'Neutral'}
            l_msg.append(d_rtn.copy())
        else:
            # if the price is not still in the book, include a new order
            d_rtn = {'agent_id': 10,
                     'instrumento_symbol': 'PETR4',
                     'order_id': my_book.i_last_order_id + 1,
                     'order_entry_step': idx,
                     'new_order_id': my_book.i_last_order_id + 1,
                     'order_price': row['Price'],
                     'order_side': row['Type'],
                     'order_status': 'New',
                     'total_qty_order': row['Size'],
                     'traded_qty_order': 0,
                     'agressor_indicator': 'Neutral'}
            l_msg.append(d_rtn)
    return l_msg


def translate_row_new(idx, row, my_book):
    '''
    Translate a line from a file of the bloomberg level I data
    :param idx: integer.
    :param row: dict.
    :param my_book: Book object.
    '''
    l_msg = []
    row['Price'] = float(row['Price'])
    row['Size'] = float(row['Size'])
    if row['Price'] == 0.:
        return l_msg
    if row['Type'] != 'TRADE' and row['Size'] % 100 == 0:
        # recover the best price
        f_best_price = my_book.get_best_price(row['Type'])
        i_order_id = my_book.i_last_order_id + 1
        # check if there is orders in the row price
        obj_ordtree = my_book.get_orders_by_price(row['Type'],
                                                  row['Price'])
        if obj_ordtree:
            # cant present more than 2 orders (mine and market)
            assert len(obj_ordtree) <= 2, 'More than two offers'
            # get the first order
            obj_order = obj_ordtree.nsmallest(1)[0][1]
            # check if should cancel the best price
            b_cancel = False
            if row['Type'] == 'BID' and row['Price'] < f_best_price:
                # check if the price in the row in smaller
                obj_ordtree2 = my_book.get_orders_by_price(row['Type'])
                best_order = obj_ordtree2.nsmallest(1)[0][1]
                d_rtn = best_order.d_msg
                d_rtn['order_status'] = 'Canceled'
                l_msg.append(d_rtn.copy())
            elif row['Type'] == 'ASK' and row['Price'] > f_best_price:
                obj_ordtree2 = my_book.get_orders_by_price(row['Type'])
                best_order = obj_ordtree2.nsmallest(1)[0][1]
                d_rtn = best_order.d_msg
                d_rtn['order_status'] = 'Canceled'
                l_msg.append(d_rtn.copy())

            # replace the current order
            i_old_id = obj_order.main_id
            i_new_id = obj_order.main_id
            if row['Size'] > obj_order['total_qty_order']:
                i_new_id = my_book.i_last_order_id + 1
                d_rtn = obj_order.d_msg
                d_rtn['order_status'] = 'Canceled'
                l_msg.append(d_rtn.copy())

            # Replace the order
            d_rtn = {'agent_id': 10,
                     'instrumento_symbol': 'PETR4',
                     'order_id': i_old_id,
                     'order_entry_step': idx,
                     'new_order_id': i_new_id,
                     'order_price': row['Price'],
                     'order_side': row['Type'],
                     'order_status': 'Replaced',
                     'total_qty_order': row['Size'],
                     'traded_qty_order': 0,
                     'agressor_indicator': 'Neutral'}
            l_msg.append(d_rtn.copy())
        else:
            # if the price is not still in the book, include a new order
            d_rtn = {'agent_id': 10,
                     'instrumento_symbol': 'PETR4',
                     'order_id': my_book.i_last_order_id + 1,
                     'order_entry_step': idx,
                     'new_order_id': my_book.i_last_order_id + 1,
                     'order_price': row['Price'],
                     'order_side': row['Type'],
                     'order_status': 'New',
                     'total_qty_order': row['Size'],
                     'traded_qty_order': 0,
                     'agressor_indicator': 'Neutral'}
            l_msg.append(d_rtn)
        return l_msg


class OrderMatching(object):
    '''
    An order matching representation that access the agents from an environment
    and handle the interation  of the individual behaviours, translating  it as
    instructions to the Order Book
    '''

    def __init__(self, env):
        '''
        Initialize a OrderMatching object. Save all parameters as attributes
        :param env: Environment object. The Market
        :param s_instrument: string. name of the instrument of book
        '''
        # save parameters as attributes
        self.env = env
        # attributes to control the qty trades by each side
        self.i_agr_ask = 0
        self.i_agr_bid = 0
        # order flow count
        self.i_ofi = 0

    def __iter__(self):
        '''
        Return the self as an iterator object. Use next() to iterate
        '''
        return self

    def next(self):
        '''
        '''
        raise NotImplementedError

    def __call__(self):
        '''
        Return the next list of messages of the simulation
        '''
        return self.next()


class BloombergMatching(OrderMatching):
    '''
    Order matching engine that use Level I data from Bloomber to reproduce the
    order book
    '''

    def __init__(self, env, s_instrument, i_num_agents, s_fname, i_idx=None):
        '''
        Initialize a OrderMatching object. Save all parameters as attributes
        :param env: Environment object. The Market
        :param s_instrument: string. name of the instrument of book
        :param env: Environment object. The Market
        '''
        super(BloombergMatching, self).__init__(env)
        self.s_instrument = s_instrument
        self.i_num_agents = i_num_agents
        self.s_fname = s_fname
        self.archive = zipfile.ZipFile(s_fname, 'r')
        self.l_fnames = self.archive.infolist()
        self.max_nfiles = len(self.l_fnames)
        self.idx = 0.
        self.i_nrow = 0.
        self.s_time = ''
        if i_idx:
            self.idx = i_idx

    def reshape_row(self, idx, row, my_book):
        '''
        Translate a line from a file of the bloomberg level I data
        :param idx: integer.
        :param row: dict.
        :param my_book: Book object.
        '''
        return translate_row(idx, row, my_book)

    def next(self):
        '''
        Return a list of messages from the agents related to the current step
        '''
        # if it will open a files that doesnt exist, stop
        if int(self.idx) > self.max_nfiles:
            raise StopIteration
        # if it is the first line of the file, open it and cerate a new book
        if self.i_nrow == 0:
            s_fname = self.l_fnames[int(self.idx)]
            self.fr_open = csv.DictReader(self.archive.open(s_fname))
            self.my_book = book.LimitOrderBook(self.s_instrument)
        # try to read a row of an already opened file
        try:
            row = self.fr_open.next()
            l_msg = self.reshape_row(self.i_nrow, row, self.my_book)
            if l_msg:
                # process each message generated by translator
                for msg in l_msg:
                    self.my_book.update(msg)
                # process the last message and use info from row
                # to compute the number of shares traded by aggressor
                if msg['order_status'] in ['Partially Filled', 'Filled']:
                    if msg['order_side'] == 'BID':
                        self.i_agr_ask += row['Size']
                    else:
                        self.i_agr_bid += row['Size']
            self.i_nrow += 1
            return l_msg
        except StopIteration:
            self.i_nrow = 0
            self.idx += 1
            # print self.my_book.get_n_top_prices(5)
            # print ""
            s_msg = 'agr Bid: {:0,.0f}, agr Ask: {:0,.0f}, TOTAL: {:0,.0f}'
            i_tot = self.i_agr_bid + self.i_agr_ask
            print s_msg.format(self.i_agr_bid, self.i_agr_ask, i_tot)
            self.i_agr_bid = 0
            self.i_agr_ask = 0
            if self.i_nrow % 1000 == 0:
                print self.i_nrow
            raise StopIteration
