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
        '''
        '''
        self.env = env

        self.quit = False
        self.start_time = None
        self.current_time = 0.0
        self.last_updated = 0.0
        self.update_delay = update_delay

        self.display = display

    def run(self, n_trials=1):
        '''
        '''
        self.quit = False
        n_trial = min(n_trials, self.env.order_book.max_nfiles)
        for trial in xrange(n_trials):
            print 'Simulator.run(): Trial {}'.format(trial)  # [debug]
            self.env.reset()
            self.current_time = 0.0
            self.last_updated = 0.0
            self.start_time = time.time()
            while True:
                try:
                    # Update current time
                    self.current_time = time.time() - self.start_time

                    # Update environment
                    f_time_step = self.current_time - self.last_updated
                    self.env.step()
                    # print information to be ploted by a visualization
                    if f_time_step >= self.update_delay:
                        pass
                        self.last_updated = self.current_time
                except StopIteration:
                    self.quit = True
                except KeyboardInterrupt:
                    self.quit = True
                finally:
                    if self.quit or self.env.done:
                        break

            if self.quit:
                break
