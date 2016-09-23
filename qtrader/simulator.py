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
    def __init__(self, env, update_delay=1.0, display=True):
        '''
        Initiate a Simulator object. Save all parameters as attributes
        Environment Object. The Environment where the agent acts
        :*param update_delay: Float. Seconds elapsed to print out the book
        :*param display: Boolean. If should open a visualizer
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
        Run the simulation
        :param n_trials: integer. Number of files to read
        '''
        n_trials = min(n_trials, self.env.order_matching.max_nfiles)
        for trial in xrange(n_trials):
            self.quit = False
            # print 'Simulator.run(): Trial {}'.format(trial + 1)  # [debug]
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
                    l_msg = self.env.step()
                    # print information to be ploted by a visualization
                    if f_time_step >= self.update_delay:
                        # TODO: Print out the scenario to be visualized
                        pass
                        self.last_updated = self.current_time
                except StopIteration:
                    self.quit = True
                except KeyboardInterrupt:
                    self.quit = True
                finally:
                    if self.quit or self.env.done:
                        break

            # if self.quit:
            #     break
