#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Implement and run an agent to learn in reinforcement learning framework

@author: ucaiado

Created on 08/18/2016
"""
import random
from environment import Agent, Environment
from simulator import Simulator
import logging
import sys
import time
from collections import defaultdict
import pandas as pd

# Log finle enabled. global variable
DEBUG = False


# setup logging messages
if DEBUG:
    s_format = '%(asctime)s;%(message)s'
    s_now = time.strftime('%c')
    s_now = s_now.replace('/', '').replace(' ', '_').replace(':', '')
    s_file = 'log/sim_{}.log'.format(s_now)
    logging.basicConfig(filename=s_file, format=s_format)
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)

    formatter = logging.Formatter(s_format)
    ch.setFormatter(formatter)
    root.addHandler(ch)


'''
Begin help functions
'''


def save_q_table(e):
    '''
    Log the final Q-table of the algorithm
    :param e: Environment object. The grid-like world
    '''
    agent = e.primary_agent
    try:
        q_table = agent.q_table
        pd.DataFrame(q_table).T.to_csv('log/qtable.log', sep='\t')
    except:
        print 'No Q-table to be printed'

'''
End help functions
'''


class BasicAgent(Agent):
    '''
    A Basic agent representation that learns to drive in the smartcab world.
    '''
    def __init__(self, env):
        '''
        Initialize a BasicLearningAgent. Save all parameters as attributes
        :param env: Environment object. The grid-like world
        '''
        # sets self.env = env, state = None, next_waypoint = None, and a
        # default color
        super(BasicAgent, self).__init__(env)
        # override color
        self.color = 'green'
        # simple route planner to get next_waypoint
        self.planner = RoutePlanner(self.env, self)
        # Initialize any additional variables here

    def reset(self, destination=None):
        '''
        Prepare for a new trip
        :param destination: tuple. the coordinates of the destination
        '''
        self.planner.route_to(destination)
        # Prepare for a new trip; reset any variables here, if required
        self.state = None
        self.next_waypoint = None
        self.old_state = None
        self.last_action = None
        self.last_reward = None

    def update(self, t):
        '''
        Update the state of the agent
        :param t: integer. Environment step attribute value
        '''
        # Gather inputs
        # from route planner, also displayed by simulator
        self.next_waypoint = self.planner.next_waypoint()
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # Update state
        self.state = self._get_intern_state(inputs, deadline)

        # Select action according to your policy
        action = self._take_action(self.state)

        # Execute action and get reward
        reward = self.env.act(self, action)

        # Learn policy based on state, action, reward
        self._apply_policy(self.state, action, reward)

        # [debug]
        s_rtn = 'LearningAgent.update(): deadline = {}, inputs = {}, action'
        s_rtn += ' = {}, reward = {}'
        if DEBUG:
            s_rtn += ', next_waypoint = {}'
            root.debug(s_rtn.format(deadline, inputs, action, reward,
                       self.next_waypoint))
        else:
            print s_rtn.format(deadline, inputs, action, reward)

    def _get_intern_state(self, inputs, deadline):
        '''
        Return a tuple representing the intern state of the agent
        :param inputs: dictionary. traffic light and presence of cars
        :param deadline: integer. time steps remaining
        '''
        return (inputs, self.next_waypoint, deadline)

    def _take_action(self, t_state):
        '''
        Return an action according to the agent policy
        :param t_state: tuple. The inputs to be considered by the agent
        '''
        return random.choice(Environment.valid_actions)

    def _apply_policy(self, state, action, reward):
        '''
        Learn policy based on state, action, reward
        :param state: dictionary. The current state of the agent
        :param action: string. the action selected at this time
        :param reward: integer. the rewards received due to the action
        '''
        pass


def run():
    """
    Run the agent for a finite number of trials.
    """
    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(BasicAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track

    # Now simulate it
    sim = Simulator(e, update_delay=0.01, display=False)
    sim.run(n_trials=100)  # run for a specified number of trials

    # save the Q table of the primary agent
    # save_q_table(e)


if __name__ == '__main__':
    # run the code
    run()
