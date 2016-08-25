#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Implement a simulator to mimic a dynamic order book environment

@author: ucaiado

Created on 08/18/2016
"""
import os
import time
import random
import importlib
import logging
import curses

# global variable
DEBUG = False

'''
Begin help functions
'''

'''
End help functions
'''


class Simulator(object):
    """
    Simulates agents in a dynamic order book environment.
    """
    def __init__(self, env, size=None, update_delay=1.0, display=True):
        self.env = env
