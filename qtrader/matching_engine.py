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
import pprint

# global variable
DEBUG = False

'''
Begin help functions
'''


class Foo(Exception):
    """
    Foo is raised by any class to help in debuging
    """
    pass


def translate_trades(idx, row, my_ordmatch, s_side=None):
    '''
    Translate trade row into trades messages. Just translate the row if the
    trade occurs at the current best price
    :param idx: integer. Order entry step
    :param row: dict. the original message from file
    :param my_ordmatch: OrderMatching object.
    :*param s_side: string. 'BID' or 'ASK'. Determine the side of the trade
    '''
    my_book = my_ordmatch.my_book
    l_msg = []
    if s_side:
        if s_side == 'BID':
            obj_price = my_ordmatch.obj_best_bid
        elif s_side == 'ASK':
            obj_price = my_ordmatch.obj_best_ask
    else:
        # check which side was affected
        # test bid side
        s_side = 'BID'
        # get the best bid price
        obj_price = my_ordmatch.obj_best_bid
        if obj_price:
            # if row['Price'] > obj_price.f_price:
            if row['Price'] != obj_price.f_price:
                obj_price = None
        # test ask side, if it is the case
        if not obj_price:
            s_side = 'ASK'
            obj_price = my_ordmatch.obj_best_ask
            if obj_price:
                # if row['Price'] < obj_price.f_price:
                if row['Price'] != obj_price.f_price:
                    return l_msg
            else:
                return l_msg
    # in case the traded qty is bigger than the price level qty, terminate
    if obj_price.i_qty < row['Size']:
        return l_msg
    # translate row in message
    i_qty = row['Size']
    for idx_ord, order_aux in obj_price.order_tree.nsmallest(1000):
        # define how much should be traded
        i_qty_traded = order_aux['org_total_qty_order']
        i_qty_traded -= order_aux['traded_qty_order']  # remain
        i_qty_traded = min(i_qty, i_qty_traded)  # minimum remain and trade
        i_qty -= i_qty_traded  # discount the traded qty
        # define the status of the message
        if order_aux['total_qty_order'] == i_qty_traded:
            s_status = 'Filled'
        else:
            s_status = 'Partially Filled'
        assert i_qty >= 0, 'Qty traded smaller than 0'
        # create the message
        i_qty2 = i_qty_traded + 1 - 1
        i_qty_traded += order_aux['traded_qty_order']
        s_action = s_side
        s_action = 'BUY'
        # if one  makes a trade at bid, it is a sell
        if s_side == 'ASK':
            s_action = 'SELL'
        d_rtn = {'agent_id': order_aux['agent_id'],
                 'instrumento_symbol': 'PETR4',
                 'order_id': order_aux['order_id'],
                 'order_entry_step': idx,
                 'new_order_id': order_aux['order_id'],
                 'order_price': row['Price'],
                 'order_side': s_side,
                 'order_status': s_status,
                 'total_qty_order': order_aux['org_total_qty_order'],
                 'traded_qty_order': i_qty_traded,
                 'agressor_indicator': 'Passive',
                 'order_qty': i_qty2,
                 'action': s_action,
                 'original_id': row['']}
        l_msg.append(d_rtn.copy())
        # create another message to update who took the action
        s_action = 'BUY'
        # if one  makes a trade at bid, it is a sell
        if s_side == 'BID':
            s_action = 'SELL'
        d_rtn = {'agent_id': order_aux['agent_id'],
                 'instrumento_symbol': 'PETR4',
                 'order_id': order_aux['order_id'],
                 'order_entry_step': idx,
                 'new_order_id': order_aux['order_id'],
                 'order_price': row['Price'],
                 'order_side': s_side,
                 'order_status': 'Filled',
                 'total_qty_order': order_aux['org_total_qty_order'],
                 'traded_qty_order': i_qty_traded,
                 'agressor_indicator': 'Agressive',
                 'order_qty': i_qty2,
                 'action': s_action,
                 'original_id': row['']}
        l_msg.append(d_rtn.copy())
    return l_msg

'''
End help functions
'''


def translate_row(idx, row, my_ordmatch, s_side=None):
    '''
    Translate a line from a file of the bloomberg level I data
    :param idx: integer. Order entry step
    :param row: dict. the original message from file
    :param my_ordmatch: OrderMatching object.
    :*param s_side: string. 'BID' or 'ASK'. Determine the side of the trade
    '''
    # reconver some variables and check if it is a valid row
    my_book = my_ordmatch.my_book
    l_msg = []
    row['Price'] = float(row['Price'])
    row['Size'] = float(row['Size'])
    if row['Price'] == 0. or row['Size'] % 100 != 0:
        return l_msg
    # update when it is a trade
    if row['Type'] == 'TRADE':
        l_msg_aux = translate_trades(idx, row, my_ordmatch, s_side)
        if len(l_msg_aux) == 0:
            return l_msg
        l_msg += l_msg_aux
    # update when it is a limit order book message
    else:
        b_replaced = False
        # recover the best price from the row side that is not just the primary
        gen_bk = ()
        if row['Type'] == 'BID':
            s_msg = 'It is a BID'
            f_best_price = my_ordmatch.best_bid[0]
            if row['Price'] <= f_best_price and f_best_price != 0:
                f_max = f_best_price + 0.01
                f_min = row['Price']
                gen_bk = my_book.book_bid.price_tree.item_slice(f_min,
                                                                f_max,
                                                                reverse=True)
        else:
            s_msg = 'It is a ASK'
            f_best_price = my_ordmatch.best_ask[0]
            if row['Price'] >= f_best_price and f_best_price != 0:
                f_max = row['Price'] + 0.01
                f_min = f_best_price
                gen_bk = my_book.book_ask.price_tree.item_slice(f_min,
                                                                f_max,
                                                                reverse=False)
        for f_price, obj_price in gen_bk:
            assert obj_price.order_tree.count <= 2, 'More than two offers'
            for idx_ord, obj_order in obj_price.order_tree.nsmallest(1000):
                # check if is the order from the primary agent
                if obj_order['agent_id'] == my_ordmatch.env.primary_agent.i_id:
                    continue
                # check if should cancel the best price
                b_cancel = False
                # check if the price in the row in smaller
                if row['Type'] == 'BID':
                    if row['Price'] < obj_order['order_price']:
                        # and cancel them
                        d_rtn = obj_order.d_msg.copy()
                        d_rtn['order_status'] = 'Canceled'
                        d_rtn['action'] = None
                        l_msg.append(d_rtn.copy())
                elif row['Type'] == 'ASK':
                    if row['Price'] > obj_order['order_price']:
                        # and cancel them
                        d_rtn = obj_order.d_msg.copy()
                        d_rtn['order_status'] = 'Canceled'
                        d_rtn['action'] = None
                        l_msg.append(d_rtn.copy())
                # replace the current order
                if row['Price'] == obj_order['order_price']:
                    i_new_id = obj_order.main_id
                    if row['Size'] > obj_order['total_qty_order']:
                        i_new_id = my_book.i_last_order_id + 1
                        d_rtn = obj_order.d_msg
                        d_rtn['order_status'] = 'Canceled'
                        d_rtn['action'] = None
                        l_msg.append(d_rtn.copy())
                    # Replace the order
                    s_action = 'BEST_BID'
                    if row['Type'] == 'ASK':
                        s_action = 'BEST_OFFER'
                    b_replaced = True
                    d_rtn = {'agent_id': 10,
                             'instrumento_symbol': 'PETR4',
                             'order_id': i_new_id,
                             'order_entry_step': idx,
                             'new_order_id': i_new_id,
                             'order_price': row['Price'],
                             'order_side': row['Type'],
                             'order_status': 'Replaced',
                             'total_qty_order': row['Size'],
                             'traded_qty_order': 0,
                             'agressor_indicator': 'Neutral',
                             'action': s_action,
                             'original_id': row['']}
                    l_msg.append(d_rtn.copy())
        if not b_replaced:
            # if the price is not still in the book, include a new order
            s_action = 'BEST_BID'
            if row['Type'] == 'ASK':
                s_action = 'BEST_OFFER'
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
                     'agressor_indicator': 'Neutral',
                     'action': s_action,
                     'original_id': row['']}
            l_msg.append(d_rtn)
    return l_msg


def translate_row_old(idx, row, my_ordmatch):
    '''
    Translate a line from a file of the bloomberg level I data
    :param idx: integer.
    :param row: dict.
    :param my_book: Book object.
    '''
    # TODO: when the row is from my agent, it should modify the order book
    # to acomodate it in a way that, when happen a trade, for instance, the
    # final state of the book is the same with my agent or not
    my_book = my_ordmatch.my_book
    l_msg = []
    row['Price'] = float(row['Price'])
    row['Size'] = float(row['Size'])
    if row['Price'] == 0. or row['Size'] % 100 != 0:
        return l_msg
    if row['Type'] == 'TRADE':
        l_msg_aux = translate_trades(idx, row, my_ordmatch)
        if len(l_msg_aux) == 0:
            return l_msg
        l_msg += l_msg_aux

    else:
        # recover the best price from the row side that is not just the primary
        # f_best_price = my_book.get_best_price(row['Type'])
        if row['Type'] == 'BID':
            f_best_price = my_ordmatch.best_bid[0]
        else:
            f_best_price = my_ordmatch.best_ask[0]
        i_order_id = my_book.i_last_order_id + 1
        # check if there are orders in the row price
        obj_ordtree = my_book.get_orders_by_price(row['Type'],
                                                  row['Price'])
        if obj_ordtree:
            # cant present more than 2 orders (mine and market)
            assert len(obj_ordtree) <= 2, 'More than two offers'
            # TODO: get the first order
            obj_order = obj_ordtree.nsmallest(1)[0][1]
            # check if should cancel the best price
            b_cancel = False
            # check if the price in the row in smaller
            if row['Type'] == 'BID' and row['Price'] < f_best_price:
                # TODO: recover orders from the best price
                obj_ordtree2 = my_book.get_orders_by_price(row['Type'])
                best_order = obj_ordtree2.nsmallest(1)[0][1]
                # and cancel them
                d_rtn = best_order.d_msg
                d_rtn['order_status'] = 'Canceled'
                d_rtn['action'] = None
                l_msg.append(d_rtn.copy())
            elif row['Type'] == 'ASK' and row['Price'] > f_best_price:
                # TODO: recover orders from the best price
                obj_ordtree2 = my_book.get_orders_by_price(row['Type'])
                best_order = obj_ordtree2.nsmallest(1)[0][1]
                # and cancel them
                d_rtn = best_order.d_msg
                d_rtn['order_status'] = 'Canceled'
                d_rtn['action'] = None
                l_msg.append(d_rtn.copy())
            # replace the current order
            i_new_id = obj_order.main_id
            if row['Size'] > obj_order['total_qty_order']:
                i_new_id = my_book.i_last_order_id + 1
                d_rtn = obj_order.d_msg
                d_rtn['order_status'] = 'Canceled'
                d_rtn['action'] = None
                l_msg.append(d_rtn.copy())
            # Replace the order
            s_action = 'BEST_BID'
            if row['Type'] == 'ASK':
                s_action = 'BEST_OFFER'
            d_rtn = {'agent_id': 10,
                     'instrumento_symbol': 'PETR4',
                     'order_id': i_new_id,
                     'order_entry_step': idx,
                     'new_order_id': i_new_id,
                     'order_price': row['Price'],
                     'order_side': row['Type'],
                     'order_status': 'Replaced',
                     'total_qty_order': row['Size'],
                     'traded_qty_order': 0,
                     'agressor_indicator': 'Neutral',
                     'action': s_action}
            l_msg.append(d_rtn.copy())
        else:
            # if the price is not still in the book, include a new order
            s_action = 'BEST_BID'
            if row['Type'] == 'ASK':
                s_action = 'BEST_OFFER'
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
                     'agressor_indicator': 'Neutral',
                     'action': s_action}
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
        self.last_date = 0
        self.best_bid = (0, 0)
        self.best_ask = (0, 0)
        self.obj_best_bid = None
        self.obj_best_ask = None
        self.i_ofi = 0
        self.i_ofi_10s = 0
        self.i_qty_traded_at_bid_10s = 0
        self.i_qty_traded_at_ask_10s = 0
        self.i_qty_traded_at_bid = 0
        self.i_qty_traded_at_ask = 0
        self.mid_price_10s = 0.
        self.b_get_new_row = True
        if i_idx:
            self.idx = i_idx

    def get_trial_identification(self):
        '''
        Return the name of the files used in the actual trial
        '''
        return self.l_fnames[int(self.idx)].filename

    def reshape_row(self, idx, row, s_side=None):
        '''
        Translate a line from a file of the bloomberg level I data
        :param idx: integer.
        :param row: dict.
        :*param s_side: string. 'BID' or 'ASK'. Determine the side of the trade
        '''
        return translate_row(idx, row, self, s_side)

    def reset(self):
        '''
        Reset the order matching and all variables needed
        '''
        # make sure that dont reset twoice
        if self.i_nrow != 0:
            self.i_nrow = 0
            self.idx += 1
            self.i_qty_traded_at_bid_10s = 0
            self.i_qty_traded_at_ask_10s = 0
            self.i_qty_traded_at_bid = 0
            self.i_qty_traded_at_ask = 0
            self.i_ofi_10s = 0
            self.i_ofi = 0
            self.last_date = 0
            self.best_bid = (0, 0)
            self.best_ask = (0, 0)
            self.obj_best_bid = None
            self.obj_best_ask = None
            self.mid_price_10s = 0.

    def next(self, b_print=False):
        '''
        Return a list of messages from the agents related to the current step
        :*param b_print: boolean. If should print the messaged generated
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
            # check if should get a new row form the file
            l_msg = []
            if self.b_get_new_row:
                row = self.fr_open.next()
                self.row = row
            else:
                print 'corrected'
                print self.my_book.get_n_top_prices(5)
                print ''
                row = self.row
                self.b_get_new_row = True
            # check if the prices have crossed themselfs
            if self.best_bid[0] != 0 and self.best_ask[0] != 0:
                if self.best_bid[0] >= self.best_ask[0]:
                    self.b_get_new_row = False
                    row_aux = row.copy()
                    row_aux['Type'] = 'TRADE'
                    row_aux['Size'] = min(self.best_ask[1], self.best_bid[1])
                    # determine a trade to this round
                    row = row_aux.copy()
                    # reshape the row to messages to order book
                    l_msg_aux = self.reshape_row(self.i_nrow, row, 'BID')
                    l_msg = self.reshape_row(self.i_nrow, row, 'ASK')
                    l_msg += l_msg_aux
                    print self.row['Date']
                    # pprint.pprint(l_msg)
                    print self.my_book.get_n_top_prices(5)
                    print ''
            # reshape the row to messages to order book when it wasnt yet
            if len(l_msg) == 0:
                # reshape the row to messages to order book
                l_msg = self.reshape_row(self.i_nrow, row)
            # measure the time in seconds
            l_aux = row['Date'].split(' ')[1].split(':')
            i_aux = sum([int(a)*60**b for a, b in zip(l_aux, [2, 1, 0])])
            self.last_date = i_aux
            # update the book
            if l_msg:
                # process each message generated by translator
                for msg in l_msg:
                    if b_print:
                        pprint.pprint(msg)
                        print ''
                    self.my_book.update(msg)
                # process the last message and use info from row
                # to compute the number of shares traded by aggressor
                if msg['order_status'] in ['Partially Filled', 'Filled']:
                    if msg['agressor_indicator'] == 'Agressive':
                        # dont process this kind of order, but keep track of
                        # the quantities traded by side
                        if msg['order_side'] == 'BID':
                            self.i_qty_traded_at_ask += msg['order_qty']
                        else:
                            self.i_qty_traded_at_bid += msg['order_qty']
            # keep the best- bid and offer in a variable
            i_bid_count = self.my_book.book_bid.price_tree.count
            i_ask_count = self.my_book.book_ask.price_tree.count
            if i_bid_count > 0 and i_ask_count > 0:
                last_bid = self.best_bid
                last_ask = self.best_ask
                o_aux = self.my_book
                best_bid = o_aux.book_bid.price_tree.max_item()
                self.obj_best_bid = best_bid[1]
                best_bid = (best_bid[0], best_bid[1].i_qty)
                best_ask = o_aux.book_ask.price_tree.min_item()
                self.obj_best_ask = best_ask[1]
                best_ask = (best_ask[0], best_ask[1].i_qty)
                # account OFI
                f_en = 0.
                if last_bid != best_bid:
                    if best_bid[0] >= last_bid[0]:
                        f_en += best_bid[1]
                    if best_bid[0] <= last_bid[0]:
                        f_en -= last_bid[1]
                if last_ask != best_ask:
                    if best_ask[0] <= last_ask[0]:
                        f_en -= best_ask[1]
                    if best_ask[0] >= last_ask[0]:
                        f_en += last_ask[1]
                self.i_ofi += f_en
                self.best_bid = best_bid
                self.best_ask = best_ask
            # hold some variables from the start of 10s fold
            if self.last_date % 10 == 0:
                self.i_ofi_10s = self.i_ofi
                self.i_ofi_10s += 1 - 1
                self.i_qty_traded_at_bid_10s = self.i_qty_traded_at_bid
                self.i_qty_traded_at_bid_10s += 1 - 1  # it is ugly
                self.i_qty_traded_at_ask_10s = self.i_qty_traded_at_ask
                self.i_qty_traded_at_ask_10s += 1 - 1
                self.mid_price_10s = (self.best_bid[0] + self.best_ask[0])/2.
            # terminate
            self.i_nrow += 1
            return l_msg
        except StopIteration:
            self.i_nrow = 0
            self.idx += 1
            self.i_qty_traded_at_bid_10s = 0
            self.i_qty_traded_at_ask_10s = 0
            self.i_qty_traded_at_bid = 0
            self.i_qty_traded_at_ask = 0
            self.i_ofi_10s = 0
            self.i_ofi = 0
            self.last_date = 0
            self.best_bid = (0, 0)
            self.best_ask = (0, 0)
            self.obj_best_bid = None
            self.obj_best_ask = None
            self.mid_price_10s = 0.
            raise StopIteration
