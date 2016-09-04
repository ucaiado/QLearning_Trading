#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Implement an Environment within all agents interact with

@author: ucaiado

Created on 08/18/2016
"""
import time
import random
from collections import OrderedDict

from matching_engine import BloombergMatching
import logging

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


'''
End help functions
'''


class Environment(object):
    '''
    Environment within which all agents operate.
    '''

    valid_actions = [None,
                     'BEST_BID',
                     'BEST_OFFER',
                     'SELL',
                     'BUY']

    def __init__(self):
        '''
        Initialize an Environment object.
        '''
        self.s_instrument = 'PETR4'
        self.done = False
        self.t = 0
        self.agent_states = OrderedDict()

        # Include Dummy agents
        self.num_dummies = 1  # no. of dummy agents
        self.last_id_agent = 10
        for i in xrange(self.num_dummies):
            self.create_agent(ZombieAgent)

        # Include Primary agent
        self.primary_agent = None  # to be set explicitly
        self.enforce_deadline = False

        # Initiate Matching Engine
        s_aux = self.s_instrument
        i_naux = self.num_dummies+1
        s_fname = 'data/petr4_0725_0818_2.zip'  # hardcoded
        self.order_matching = BloombergMatching(env=self,
                                                s_instrument=s_aux,
                                                i_num_agents=i_naux,
                                                s_fname=s_fname)

    def create_agent(self, agent_class, *args, **kwargs):
        '''
        Include a agent in the environment and initiate its env state
        :param agent_class: Agent Object. The agent desired
        :*param args, kwargs: any type. Any other parameter needed by the agent
        '''

        kwargs['i_id'] = self.last_id_agent
        agent = agent_class(self, *args, **kwargs)
        self.agent_states[agent] = {'qBid': 0,
                                    'Bid': 0.,
                                    'Ask': 0.,
                                    'qAsk': 0,
                                    'Position': 0,
                                    'Agent': agent,
                                    'best_bid': False,
                                    'best_offer': False}
        self.last_id_agent += 1
        return agent

    def set_primary_agent(self, agent):
        '''
        Initiate the agent that is supposed to be modeled
        :param agent: Agent Object. The agent used as primary
        '''
        self.primary_agent = agent
        self.agent_states[agent] = {'qBid': 0,
                                    'Bid': 0.,
                                    'Ask': 0.,
                                    'qAsk': 0,
                                    'Position': 0,
                                    'Agent': agent,
                                    'best_bid': False,
                                    'best_offer': False}

    def reset(self):
        '''
        Reset the environment and all variables needed as well as the states
        of each agent
        '''
        self.done = False
        self.t = 0

        # reset environment
        s_msg = 'Environment.reset(): Trial set up to use {} file'
        s_name = self.order_matching.get_trial_identification()
        if DEBUG:
            logging.info(s_msg.format(s_name))
        else:
            print s_msg.format(s_name)

        # Initialize agent(s)
        for agent in self.agent_states.iterkeys():
            self.agent_states[agent] = {'qBid': 0,
                                        'Bid': 0.,
                                        'Ask': 0.,
                                        'qAsk': 0,
                                        'Position': 0,
                                        'Agent': agent,
                                        'best_bid': False,
                                        'best_offer': False}
            agent.reset()

    def step(self):
        '''
        Perform a discreate step in the environment updating the state of all
        agents
        '''
        # Update agents asking to the order matching what each one has done
        l_msg = self.order_matching.next()
        i_time = self.order_matching.last_date
        for msg in l_msg:
            agent_aux = self.agent_states[msg['agent_id']]['Agent']
            self.update_agent_state(agent=agent_aux, i_time=i_time, msg=msg)

        self.t += 1

    def sense(self, agent):
        '''
        '''
        assert agent in self.agent_states, 'Unknown agent!'

        state = self.agent_states[agent]
        location = state['location']
        heading = state['heading']
        light = 'green' if (self.intersections[location].state and heading[1] != 0) or ((not self.intersections[location].state) and heading[0] != 0) else 'red'

        # Populate oncoming, left, right
        oncoming = None
        left = None
        right = None
        for other_agent, other_state in self.agent_states.iteritems():
            if agent == other_agent or location != other_state['location'] or (heading[0] == other_state['heading'][0] and heading[1] == other_state['heading'][1]):
                continue
            other_heading = other_agent.get_next_waypoint()
            if (heading[0] * other_state['heading'][0] + heading[1] * other_state['heading'][1]) == -1:
                if oncoming != 'left':  # we don't want to override oncoming == 'left'
                    oncoming = other_heading
            elif (heading[1] == other_state['heading'][0] and -heading[0] == other_state['heading'][1]):
                if right != 'forward' and right != 'left':  # we don't want to override right == 'forward or 'left'
                    right = other_heading
            else:
                if left != 'forward':  # we don't want to override left == 'forward'
                    left = other_heading

        # TODO: make this a namedtuple
        return {'light': light, 'oncoming': oncoming, 'left': left, 'right': right}

    def act(self, agent, action):
        '''
        Return the environment reward or penalty by the agent's action and
        current state. Also, update the known condition of the agent's state
        by the Environment
        :param agent: Agent object. the agent that will perform the action
        :param action: dictionary. The current action of the agent
        '''
        assert agent in self.agent_states, 'Unknown agent!'
        assert action['action'] in self.valid_actions, 'Invalid action!'

        # Update the position using action
        agent.act(action)
        position = agent.position
        # update current position in the agent state
        state = self.agent_states[agent]
        for s_key in position:
            state[s_key] = position[s_key]
        state['Position'] = state['qBid'] - state['qAsk']
        # measure the reward

        reward = 0.

        # location = state['location']
        # heading = state['heading']
        # light = 'green' if (self.intersections[location].state and heading[1] != 0) or ((not self.intersections[location].state) and heading[0] != 0) else 'red'
        # sense = self.sense(agent)

        # # Move agent if within bounds and obeys traffic rules
        # reward = 0  # reward/penalty
        # move_okay = True
        # if action == 'forward':
        #     if light != 'green':
        #         move_okay = False
        # elif action == 'left':
        #     if light == 'green' and (sense['oncoming'] == None or sense['oncoming'] == 'left'):
        #         heading = (heading[1], -heading[0])
        #     else:
        #         move_okay = False
        # elif action == 'right':
        #     if light == 'green' or sense['left'] != 'straight':
        #         heading = (-heading[1], heading[0])
        #     else:
        #         move_okay = False

        # if move_okay:
        #     # Valid move (could be null)
        #     if action is not None:
        #         # Valid non-null move
        #         location = ((location[0] + heading[0] - self.bounds[0]) % (self.bounds[2] - self.bounds[0] + 1) + self.bounds[0],
        #                     (location[1] + heading[1] - self.bounds[1]) % (self.bounds[3] - self.bounds[1] + 1) + self.bounds[1])  # wrap-around
        #         #if self.bounds[0] <= location[0] <= self.bounds[2] and self.bounds[1] <= location[1] <= self.bounds[3]:  # bounded
        #         state['location'] = location
        #         state['heading'] = heading
        #         reward = 2.0 if action == agent.get_next_waypoint() else -0.5  # valid, but is it correct? (as per waypoint)
        #     else:
        #         # Valid null move
        #         reward = 0.0
        # else:
        #     # Invalid move
        #     reward = -1.0

        # if agent is self.primary_agent:
        #     if state['location'] == state['destination']:
        #         if state['deadline'] >= 0:
        #             reward += 10  # bonus
        #         self.done = True

        #         # change this part of the code to log the messages to a file
        #         s_msg = 'Environment.act(): Primary agent has reached destination!'  # [debug]
        #         if DEBUG:
        #             logging.info(s_msg)
        #         else:
        #             print s_msg
        #     self.status_text = 'state: {}\naction: {}\nreward: {}'.format(agent.get_state(), action, reward)
        #     #print 'Environment.act() [POST]: location: {}, heading: {}, action: {}, reward: {}'.format(location, heading, action, reward)  # [debug]

        return reward

    def update_agent_state(self, agent, i_time, msg):
        '''
        Update the agent state dictionary
        :param agent: Agent Object. The agent used as primary
        :param msg: string. Order matching message
        :param t: integer. Time stemp of the order book
        '''
        assert agent in self.agent_states, 'Unknown agent!'
        agent.update(s_msg=msg, i_time=i_time)
        for s_key in ['qBid', 'Bid', 'Ask', 'qAsk']:
            self.agent_states[agent][s_key] = agent[s_key]
        qBid = self.agent_states[agent]['qBid']
        qAsk = self.agent_states[agent]['qAsk']
        self.agent_states[agent]['Position'] = qBid - qAsk

    def get_order_book(self):
        '''
        Return a dataframe with the first 5 levels of the current order book
        '''
        return self.order_matching.my_book.get_n_top_prices(5)


class Agent(object):
    '''
    Base class for all agents.
    '''
    # dict to use to find out what side the book was traded by the agent
    trade_side = {'Agressive': {'BID': 'Ask', 'ASK': 'Bid'},
                  'Passive': {'BID': 'Bid', 'ASK': 'Ask'}}

    def __init__(self, env, i_id):
        '''
        Initiate an Agent object. Save all parameters as attributes
        :param env: Environment Object. The Environment where the agent acts
        :param i_id: integer. Agent id
        '''
        self.env = env
        self.i_id = i_id
        self.state = None
        self.position = {'qAsk': 0, 'Ask': 0., 'qBid': 0, 'Bid': 0.}
        self.d_order_map = {}

    def reset(self):
        '''
        Reset the state and the agent's memory about its positions
        '''
        self.state = None
        self.position = {'qAsk': 0, 'Ask': 0., 'qBid': 0, 'Bid': 0.}
        self.d_order_map = {}

    def act(self, msg):
        '''
        Update the positions of the agent based on the message passed
        :param msg: string. Order matching message
        '''
        # recover some variables to use
        s_status = msg['order_status']
        i_id = msg['order_id']
        # update position
        if s_status in ['New', 'Replaced']:
            self.d_order_map[i_id] = msg
        if s_status in ['Canceled', 'Expired']:
            self.d_order_map.pop(i_id)
        elif s_status in ['Filled', 'Partially Filled']:
            # update the order map, if it was a passive trade
            if msg['agressor_indicator'] == 'Passive':
                if s_status == 'Filled':
                    self.d_order_map.pop(i_id)
                else:
                    self.d_order_map[i_id] = msg
            # account the trades
            s_tside = self.trade_side[msg['agressor_indicator']]
            s_tside = s_tside[msg['order_side']]
            self.position['q' + s_tside] += msg['order_qty']
            f_volume = msg['order_price'] * msg['order_qty']
            self.position[s_tside] += f_volume

    def update(self, s_msg, i_time):
        '''
        Update the inner state of the agent
        :param s_msg: string. Order matching message
        :param i_time: integer. Time stemp of the order book
        '''
        NotImplementedError('This class should be implemented')

    def get_state(self):
        '''
        Return the inner state of the agent
        '''
        return self.state

    def get_position(self):
        '''
        Return the positions of the agent
        '''
        return self.position

    def _get_intern_state(self, inputs, position):
        '''
        Return a tuple representing the intern state of the agent
        :param inputs: dictionary. traffic light and presence of cars
        :param position: dictionary. the current position of the agent
        '''
        pass

    def _take_action(self, t_state, s_msg):
        '''
        Return an action according to the agent policy
        :param t_state: tuple. The inputs to be considered by the agent
        :param s_msg: string. Order matching message
        '''
        return s_msg

    def _apply_policy(self, state, action, reward):
        '''
        Learn policy based on state, action, reward
        :param state: dictionary. The current state of the agent
        :param action: string. the action selected at this time
        :param reward: integer. the rewards received due to the action
        '''
        pass

    def __str__(self):
        '''
        Return the name of the Agent
        '''
        return str(self.i_id)

    def __repr__(self):
        '''
        Return the name of the Agent
        '''
        return str(self.i_id)

    def __eq__(self, other):
        '''
        Return if a Agent has equal i_id from the other
        :param other: agent object. Agent to be compared
        '''
        i_id = other
        if not isinstance(other, int):
            i_id = other.i_id
        return self.i_id == i_id

    def __ne__(self, other):
        '''
        Return if a Agent has different i_id from the other
        :param other: agent object. Agent to be compared
        '''
        return not self.__eq__(other)

    def __hash__(self):
        '''
        Allow the Agent object be used as a key in a hash table. It is used by
        dictionaries
        '''
        return self.i_id.__hash__()

    def __getitem__(self, s_idx):
        '''
        Allow direct access to the position information of the object
        '''
        return self.position[s_idx]


class ZombieAgent(Agent):
    '''
    A ZombieAgent just obeys what the order matching engine determines
    '''

    def __init__(self, env, i_id):
        '''
        Initiate a ZombieAgent object. save all parameters as attributes
        :param env: Environment Object. The Environment where the agent acts
        :param i_id: integer. Agent id
        '''
        super(ZombieAgent, self).__init__(env, i_id)

    def update(self, s_msg, i_time):
        '''
        Update the state of the agent
        :param s_msg: string. A message generated by the order matching
        :param i_time: integer. The current time of the order book
        '''
        # self.act(s_msg)
        # inputs = self.env.sense(self)

        # action = None
        # if action_okay:
        #     action = self.next_waypoint
        #     self.next_waypoint = random.choice(Environment.valid_actions[1:])
        # reward = self.env.act(self, action)

        # inputs = self.env.sense(self)
        inputs = {}  # TODO: correct that
        state = self.env.agent_states[self]

        # Update state (position ,volume and if keep an order in bid or ask)
        self.state = self._get_intern_state(inputs, state)

        # Select action according to the agent's policy
        action = self._take_action(self.state, s_msg)

        # # Execute action and get reward
        # print '\ncurrent action: {}\n'.format(action)
        reward = self.env.act(self, action)

        # Learn policy based on state, action, reward
        self._apply_policy(self.state, action, reward)

        # [debug]
        s_rtn = 'ZombieAgent.update(): position = {}, inputs = {}, action'
        s_rtn += ' = {}, reward = {}'
        # if DEBUG:
        #     root.debug(s_rtn.format(state['Position'],
        #                             inputs,
        #                             action,
        #                             reward))
        # else:
        #     print s_rtn.format(state['Position'],
        #                        inputs,
        #                        action['action'],
        #                        reward)
