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
from simulator import Simulator
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
    """
    Environment within which all agents operate.
    """

    valid_actions = [None, 'forward', 'left', 'right']
    valid_inputs = {'light': TrafficLight.valid_states,
                    'oncoming': valid_actions,
                    'left': valid_actions,
                    'right': valid_actions}
    valid_headings = [(1, 0), (0, -1), (-1, 0), (0, 1)]  # ENWS
    # even if enforce_deadline is False, end trial when deadline reaches this
    # value (to avoid deadlocks)
    hard_time_limit = -100

    def __init__(self):
        self.done = False


class Agent(object):
    """
    Base class for all agents.
    """

    def __init__(self, env):
        self.env = env
        self.state = None
        self.next_waypoint = None
        self.color = 'cyan'

    def reset(self, destination=None):
        pass

    def update(self, t):
        pass

    def get_state(self):
        return self.state

    def get_next_waypoint(self):
        return self.next_waypoint
