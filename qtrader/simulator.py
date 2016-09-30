#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Implement a simulator to mimic a dynamic order book environment

@author: ucaiado

Created on 08/18/2016
"""
import importlib
import logging
import os
import pandas as pd
import random
import time


# global variable
DEBUG = True

'''
Begin help functions
'''


def save_q_table(e, i_trial):
    '''
    Log the final Q-table of the algorithm
    :param e: Environment object. The order book
    :param i_trial: integer. id of the current trial
    '''
    agent = e.primary_agent
    try:
        q_table = agent.q_table
        # define the name of the files
        s_fname = 'log/qtable/{}_qtable_{}.log'
        s_fname = s_fname.format(agent.s_agent_name, i_trial)
        # save data structures
        pd.DataFrame(q_table).T.to_csv(s_fname, sep='\t')
    except:
        print 'No Q-table to be printed'

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

    def train(self, n_trials=1, n_sessions=1):
        '''
        Run the simulation to train the algorithm
        :*param n_sessions: integer. Number of files to read
        :*param n_trials: integer. Iterations over the same files
        '''
        n_sessions = min(n_sessions, self.env.order_matching.max_nfiles)

        for trial in xrange(n_trials):
            # reset the order matching to the initial point
            self.env.reset_order_matching_idx()
            for i_sess in xrange(n_sessions):
                self.quit = False
                # [debug]
                # print 'Simulator.run(): Trial {}'.format(trial + 1)
                self.env.reset()
                self.current_time = 0.0
                self.last_updated = 0.0
                self.start_time = time.time()
                # iterate over the current dataset
                while True:
                    try:
                        # Update current time
                        self.current_time = time.time() - self.start_time
                        # Update environment
                        f_time_step = self.current_time - self.last_updated
                        l_msg = self.env.step()
                        # print information to be used by a visualization
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
                # save the current Q-table
                save_q_table(self.env, trial+1)

                # if self.quit:
                #     break

    def test(self, s_qtable, n_trials=1, n_sessions=1, i_idx=None):
        '''
        Run the simulation to test the policy learned
        :param s_qtable: string. path to the qtable to be used
        :*param n_sessions: integer. Number of files to read
        :*param n_trials: integer. Iterations over the same files
        :*param i_idx: integer. start file of the envioronment
        '''
        n_sessions = min(n_sessions, self.env.order_matching.max_nfiles)
        agent = self.env.primary_agent
        if agent.s_agent_name != 'BasicAgent':
            agent.set_qtable(s_qtable)

        for trial in xrange(n_trials):
            # reset the order matching to the initial point
            self.env.reset_order_matching_idx(i_idx=i_idx)
            for i_sess in xrange(n_sessions):
                self.quit = False
                # [debug]
                # print 'Simulator.run(): Trial {}'.format(trial + 1)
                self.env.reset()
                self.current_time = 0.0
                self.last_updated = 0.0
                self.start_time = time.time()
                # iterate over the current dataset
                while True:
                    try:
                        # Update current time
                        self.current_time = time.time() - self.start_time
                        # Update environment
                        f_time_step = self.current_time - self.last_updated
                        l_msg = self.env.step()
                        # print information to be used by a visualization
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

    def in_sample_test(self, n_trials=1, n_sessions=1):
        '''
        Test the performance of the different policies learned after each trial
        across the same datset used to create them
        :*param n_sessions: integer. Number of files to read
        :*param n_trials: integer. Iterations over the same files
        '''
        agent = self.env.primary_agent
        for trial in xrange(n_trials):
            s_qtable = 'log/qtable/{}_qtable_{}.log'
            s_qtable = s_qtable.format(agent.s_agent_name, trial+1)
            self.test(s_qtable=s_qtable,
                      n_trials=1,
                      n_sessions=n_sessions)

    def out_of_sample(self, s_qtable, n_start, n_trials=1, n_sessions=1):
        '''
        Test the performance of the a given policy starting on the files index
        passed as parameter
        :param s_qtable: string. path to the qtable to be used
        :param n_start: integer. start file to use in simulation
        :*param n_sessions: integer. Number of files to read
        :*param n_trials: integer. Iterations over the same files
        '''
        agent = self.env.primary_agent
        for trial in xrange(n_trials):
            s_qtable = s_qtable.format(agent.s_agent_name, trial+1)
            self.test(s_qtable=s_qtable,
                      n_trials=1,
                      n_sessions=n_sessions,
                      i_idx=n_start)
