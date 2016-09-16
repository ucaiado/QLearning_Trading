#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Translate messages from files and from agents to the matching engine

@author: ucaiado

Created on 09/16/2016
"""


def translate_trades(idx, row, my_ordmatch, s_side=None, i_id=None):
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
            if row['Price'] > obj_price.f_price:
                obj_price = None
        # test ask side, if it is the case
        if not obj_price:
            s_side = 'ASK'
            obj_price = my_ordmatch.obj_best_ask
            if obj_price:
                if row['Price'] < obj_price.f_price:
                    return l_msg
            else:
                return l_msg
    # in case the traded qty is bigger than the price level qty, terminate
    if obj_price.i_qty < row['Size']:
        return l_msg
    # translate row in message
    i_qty = row['Size']
    for idx_ord, order_aux in obj_price.order_tree.nsmallest(1000):
        # check the id of the aggressor
        if not i_id:
            i_agrr = 10
        else:
            i_agrr = i_id
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
        # check the id of the aggressive side

        # create another message to update who took the action
        s_action = 'BUY'
        # if one  makes a trade at bid, it is a sell
        if s_side == 'BID':
            s_action = 'SELL'
        d_rtn = {'agent_id': i_agrr,
                 'instrumento_symbol': 'PETR4',
                 'order_id': my_book.i_last_order_id + 1,
                 'order_entry_step': idx,
                 'new_order_id': my_book.i_last_order_id + 1,
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
                if my_ordmatch.env.primary_agent:
                    i_primary_id = my_ordmatch.env.primary_agent.i_id
                    if obj_order['agent_id'] == i_primary_id:
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


def translate_to_agent(agent, s_action, my_ordmatch):
    '''
    Translate a line from a file of the bloomberg level I data. Is expected
    that the agent has one orders by side, at maximum
    :param agent: Agent Object.
    :param s_action: string.
    :param my_ordmatch: OrderMatching object.
    '''
    # reconver some variables and check if it is a valid row
    my_book = my_ordmatch.my_book
    l_msg = []
    # recover the best price from the row side that is not just the primary
    t_best_bid = my_ordmatch.best_bid
    t_best_ask = my_ordmatch.best_ask
    my_order_bid = None
    my_order_ask = None
    if agent.d_order_tree['BID'].count > 0:
        # get the maximum price
        f_bid, my_order_bid = agent.d_order_tree['BID'].max_item()
    if agent.d_order_tree['ASK'].count > 0:
        # get the minimum price
        f_ask, my_order_ask = agent.d_order_tree['ASK'].min_item()
    # check if do nothing or cancel all
    if not s_action:
        # check if should cancel all
        if my_order_bid:
            # and cancel them
            d_rtn = my_order_bid.copy()
            d_rtn['order_status'] = 'Canceled'
            d_rtn['action'] = s_action
            l_msg.append(d_rtn.copy())
        if my_order_ask:
            # and cancel them
            d_rtn = my_order_ask.copy()
            d_rtn['order_status'] = 'Canceled'
            d_rtn['action'] = s_action
            l_msg.append(d_rtn.copy())
        return l_msg
    # update when it has a limit order book message related to the bid side
    if s_action in ['BEST_BID', 'BEST_BOTH']:
        # cancel ask side
        if my_order_ask:
            if s_action == 'BEST_BID':
                # and cancel them
                d_rtn = my_order_ask.copy()
                d_rtn['order_status'] = 'Canceled'
                d_rtn['action'] = s_action
                l_msg.append(d_rtn.copy())
        # check if should change the price
        if my_order_bid:
            # cancel the old order
            if my_order_bid['order_price'] != t_best_bid[0]:
                # and cancel them
                d_rtn = my_order_bid.copy()
                d_rtn['order_status'] = 'Canceled'
                d_rtn['action'] = s_action
                l_msg.append(d_rtn.copy())
                # replace it with a new ID
                d_rtn = {'agent_id': agent.i_id,
                         'instrumento_symbol': 'PETR4',
                         'order_id': my_book.i_last_order_id + 1,
                         'order_entry_step': my_ordmatch.i_nrow,
                         'new_order_id': my_book.i_last_order_id + 1,
                         'order_price': t_best_bid[0] - 0.10,
                         'order_side': 'BID',
                         'order_status': 'Replaced',
                         'total_qty_order': 100,
                         'traded_qty_order': 0,
                         'agressor_indicator': 'Neutral',
                         'action': s_action,
                         'original_id': -1}
                my_book.i_last_order_id += 1
                l_msg.append(d_rtn.copy())
        else:
            # include a new order
            d_rtn = {'agent_id': agent.i_id,
                     'instrumento_symbol': 'PETR4',
                     'order_id': my_book.i_last_order_id + 1,
                     'order_entry_step': my_ordmatch.i_nrow,
                     'new_order_id': my_book.i_last_order_id + 1,
                     'order_price': t_best_bid[0] - 0.10,
                     'order_side': 'BID',
                     'order_status': 'New',
                     'total_qty_order': 100,
                     'traded_qty_order': 0,
                     'agressor_indicator': 'Neutral',
                     'action': s_action,
                     'original_id': -1}
            my_book.i_last_order_id += 1
            l_msg.append(d_rtn.copy())
    # update when it has a limit order book message related to the ask side
    if s_action in ['BEST_ASK', 'BEST_BOTH']:
        # cancel ask side
        if my_order_bid:
            if s_action == 'BEST_ASK':
                # and cancel them
                d_rtn = my_order_bid.d_msg.copy()
                d_rtn['order_status'] = 'Canceled'
                d_rtn['action'] = s_action
                l_msg.append(d_rtn.copy())
        # check if should change the price
        if my_order_ask:
            # cancel the old order
            if my_order_ask['order_price'] != t_best_ask[0]:
                # and cancel them
                d_rtn = my_order_ask.d_msg.copy()
                d_rtn['order_status'] = 'Canceled'
                d_rtn['action'] = s_action
                l_msg.append(d_rtn.copy())
                # replace it with a new ID
                d_rtn = {'agent_id': agent.i_id,
                         'instrumento_symbol': 'PETR4',
                         'order_id': my_book.i_last_order_id + 1,
                         'order_entry_step': my_ordmatch.i_nrow,
                         'new_order_id': my_book.i_last_order_id + 1,
                         'order_price': t_best_ask[0] + 0.10,
                         'order_side': 'ASK',
                         'order_status': 'Replaced',
                         'total_qty_order': 100,
                         'traded_qty_order': 0,
                         'agressor_indicator': 'Neutral',
                         'action': s_action,
                         'original_id': -1}
                my_book.i_last_order_id += 1
                l_msg.append(d_rtn.copy())
        else:
            # include a new order
            d_rtn = {'agent_id': agent.i_id,
                     'instrumento_symbol': 'PETR4',
                     'order_id': my_book.i_last_order_id + 1,
                     'order_entry_step': my_ordmatch.i_nrow,
                     'new_order_id': my_book.i_last_order_id + 1,
                     'order_price': t_best_ask[0] + 0.10,
                     'order_side': 'ASK',
                     'order_status': 'New',
                     'total_qty_order': 100,
                     'traded_qty_order': 0,
                     'agressor_indicator': 'Neutral',
                     'action': s_action,
                     'original_id': -1}
            my_book.i_last_order_id += 1
            l_msg.append(d_rtn.copy())

    return l_msg
