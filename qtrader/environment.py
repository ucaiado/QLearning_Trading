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

import matching_engine
import logging

# global variable
DEBUG = False

'''
Begin help functions
'''

'''
End help functions
'''


class Environment(object):
    '''
    Environment within which all agents operate.
    '''

    valid_actions = [None, 'BID', 'ASK', 'TRADE']
    valid_headings = [(1, 0), (0, -1), (-1, 0), (0, 1)]  # ENWS
    # even if enforce_deadline is False, end trial when deadline reaches this
    # value (to avoid deadlocks)
    hard_time_limit = -100

    def __init__(self):
        '''
        Initialize a Environment object.
        '''
        self.s_instrument = 'FOO'
        self.done = False
        self.t = 0
        self.agent_states = OrderedDict()
        self.status_text = ''

        # Dummy agents
        self.num_dummies = 1  # no. of dummy agents
        for i in xrange(self.num_dummies):
            self.create_agent(DummyAgent)

        # Primary agent
        self.primary_agent = None  # to be set explicitly
        self.enforce_deadline = False

        # Matching Engine
        s_aux = self.s_instrument
        i_naux = self.num_dummies+1
        self.order_matching = matching_engine.BloombergMatching(self,
                                                                s_aux,
                                                                i_naux,
                                                                s_fname)

    def create_agent(self, agent_class, *args, **kwargs):
        '''
        Include a agent in the environment and initiate its env state
        '''
        agent = agent_class(self, *args, **kwargs)
        self.agent_states[agent] = {'qBid': 0.,
                                    'vBid': 0.,
                                    'qAsk': 0.,
                                    'vAsk': 0.,
                                    'Position': 0.}
        return agent

    def set_primary_agent(self, agent, enforce_deadline=False):
        '''
        Initiate the agent that is suposed to be modeled
        '''
        self.primary_agent = agent
        self.enforce_deadline = enforce_deadline

    def reset(self):
        '''
        Reset the enrironment and reinitialize all variables needed
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
            self.agent_states[agent] = {'qBid': 0.,
                                        'vBid': 0.,
                                        'qAsk': 0.,
                                        'vAsk': 0.,
                                        'Position': 0.}
            agent.reset()

    def step(self):
        '''
        Perform a discreate step in the environment updating the state of all
        agents
        '''
        # Update agents asking to the order matching what each one has done
        l_msg = self.order_book
        for agent in self.agent_states.iterkeys():
            agent.update(self.t)

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

        return {'light': light, 'oncoming': oncoming, 'left': left, 'right': right}  # TODO: make this a namedtuple

    def get_deadline(self, agent):
        '''
        '''
        return self.agent_states[agent]['deadline'] if agent is self.primary_agent else None

    def act(self, agent, action):
        '''
        '''
        assert agent in self.agent_states, 'Unknown agent!'
        assert action in self.valid_actions, 'Invalid action!'

        state = self.agent_states[agent]
        location = state['location']
        heading = state['heading']
        light = 'green' if (self.intersections[location].state and heading[1] != 0) or ((not self.intersections[location].state) and heading[0] != 0) else 'red'
        sense = self.sense(agent)

        # Move agent if within bounds and obeys traffic rules
        reward = 0  # reward/penalty
        move_okay = True
        if action == 'forward':
            if light != 'green':
                move_okay = False
        elif action == 'left':
            if light == 'green' and (sense['oncoming'] == None or sense['oncoming'] == 'left'):
                heading = (heading[1], -heading[0])
            else:
                move_okay = False
        elif action == 'right':
            if light == 'green' or sense['left'] != 'straight':
                heading = (-heading[1], heading[0])
            else:
                move_okay = False

        if move_okay:
            # Valid move (could be null)
            if action is not None:
                # Valid non-null move
                location = ((location[0] + heading[0] - self.bounds[0]) % (self.bounds[2] - self.bounds[0] + 1) + self.bounds[0],
                            (location[1] + heading[1] - self.bounds[1]) % (self.bounds[3] - self.bounds[1] + 1) + self.bounds[1])  # wrap-around
                #if self.bounds[0] <= location[0] <= self.bounds[2] and self.bounds[1] <= location[1] <= self.bounds[3]:  # bounded
                state['location'] = location
                state['heading'] = heading
                reward = 2.0 if action == agent.get_next_waypoint() else -0.5  # valid, but is it correct? (as per waypoint)
            else:
                # Valid null move
                reward = 0.0
        else:
            # Invalid move
            reward = -1.0

        if agent is self.primary_agent:
            if state['location'] == state['destination']:
                if state['deadline'] >= 0:
                    reward += 10  # bonus
                self.done = True

                # change this part of the code to log the messages to a file
                s_msg = 'Environment.act(): Primary agent has reached destination!'  # [debug]
                if DEBUG:
                    logging.info(s_msg)
                else:
                    print s_msg
            self.status_text = 'state: {}\naction: {}\nreward: {}'.format(agent.get_state(), action, reward)
            #print 'Environment.act() [POST]: location: {}, heading: {}, action: {}, reward: {}'.format(location, heading, action, reward)  # [debug]

        return reward

    def compute_dist(self, a, b):
        '''
        L1 distance between two points.
        '''
        return abs(b[0] - a[0]) + abs(b[1] - a[1])


class Agent(object):
    '''
    Base class for all agents.
    '''

    def __init__(self, env, i_id):
        '''
        Initiate an Agent object. Save all parameters as attributes
        :param env: Environment Object. The environment that the agent interact
        :param i_id: integer. Agent id
        '''
        self.env = env
        self.state = None
        self.qBid = 0
        self.qAsk = 0
        self.Bid = 0
        self.Ask = 0

    def reset(self, destination=None):
        '''
        '''
        self.qBid = 0
        self.qAsk = 0
        self.Bid = 0
        self.Ask = 0

    def act(self):
        '''
        '''
        pass

    def update(self, t):
        '''
        '''
        pass

    def get_state(self):
        '''
        '''
        return self.state

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
        return self.i_id == other.i_id

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


class DummyAgent(Agent):
    '''
    '''

    def __init__(self, env):
        '''
        '''

        # sets self.env = env, state = None, next_waypoint = None, and a
        # default color
        super(DummyAgent, self).__init__(env)
        self.next_waypoint = random.choice(Environment.valid_actions[1:])

    def update(self, s_msg):
        '''
        Update the state of the agent
        :param s_msg: string. A message generated by the order matching
        '''
        inputs = self.env.sense(self)



        action = None
        if action_okay:
            action = self.next_waypoint
            self.next_waypoint = random.choice(Environment.valid_actions[1:])
        reward = self.env.act(self, action)
