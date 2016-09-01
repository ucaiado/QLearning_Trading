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
    valid_inputs = {'light': TrafficLight.valid_states,
                    'oncoming': valid_actions,
                    'left': valid_actions,
                    'right': valid_actions}
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
        '''
        agent = agent_class(self, *args, **kwargs)
        self.agent_states[agent] = {'location': 0,
                                    'heading': (0, 1)}
        return agent

    def set_primary_agent(self, agent, enforce_deadline=False):
        '''
        '''
        self.primary_agent = agent
        self.enforce_deadline = enforce_deadline

    def reset(self):
        '''
        '''
        self.done = False
        self.t = 0

        # Reset traffic lights
        # for traffic_light in self.intersections.itervalues():
        #     traffic_light.reset()

        # Pick a start and a destination
        start = random.choice(self.intersections.keys())
        destination = random.choice(self.intersections.keys())

        # Ensure starting location and destination are not too close
        while self.compute_dist(start, destination) < 4:
            start = random.choice(self.intersections.keys())
            destination = random.choice(self.intersections.keys())

        start_heading = random.choice(self.valid_headings)
        deadline = self.compute_dist(start, destination) * 5
        s_msg = 'Environment.reset(): Trial set up with start = {}, '
        s_msg += 'destination = {}, deadline = {}'
        if DEBUG:
            logging.info(s_msg.format(start, destination, deadline))
        else:
            print s_msg.format(start, destination, deadline)

        # Initialize agent(s)
        for agent in self.agent_states.iterkeys():
            self.agent_states[agent] = {
                'location': start if agent is self.primary_agent else random.choice(self.intersections.keys()),
                'heading': start_heading if agent is self.primary_agent else random.choice(self.valid_headings),
                'destination': destination if agent is self.primary_agent else None,
                'deadline': deadline if agent is self.primary_agent else None}
            agent.reset(destination=(destination if agent is self.primary_agent else None))

    def step(self):
        '''
        '''
        # print "Environment.step(): t = {}".format(self.t)  # [debug]

        # Update traffic lights
        # for intersection, traffic_light in self.intersections.iteritems():
        #     traffic_light.update(self.t)

        # Update agents asking to the order matching what each one has done
        l_msg = self.order_book
        # for agent in self.agent_states.iterkeys():
        #     agent.update(self.t)

        self.t += 1
        if self.primary_agent is not None:
            agent_deadline = self.agent_states[self.primary_agent]['deadline']

            # change this part of the code to log the messages to a file
            s_msg = None
            if agent_deadline <= self.hard_time_limit:
                self.done = True
                s_msg = "Environment.step(): Primary agent hit hard time limit ({})! Trial aborted.".format(self.hard_time_limit)
            elif self.enforce_deadline and agent_deadline <= 0:
                self.done = True
                s_msg = "Environment.step(): Primary agent ran out of time! Trial aborted."
            if s_msg:
                if DEBUG:
                    logging.info(s_msg)
                else:
                    print s_msg

            self.agent_states[self.primary_agent]['deadline'] = agent_deadline - 1

    def sense(self, agent):
        '''
        '''
        assert agent in self.agent_states, "Unknown agent!"

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
        assert agent in self.agent_states, "Unknown agent!"
        assert action in self.valid_actions, "Invalid action!"

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
                s_msg = "Environment.act(): Primary agent has reached destination!"  # [debug]
                if DEBUG:
                    logging.info(s_msg)
                else:
                    print s_msg
            self.status_text = "state: {}\naction: {}\nreward: {}".format(agent.get_state(), action, reward)
            #print "Environment.act() [POST]: location: {}, heading: {}, action: {}, reward: {}".format(location, heading, action, reward)  # [debug]

        return reward

    def compute_dist(self, a, b):
        '''
        '''
        """L1 distance between two points."""
        return abs(b[0] - a[0]) + abs(b[1] - a[1])


class Agent(object):
    '''
    Base class for all agents.
    '''

    def __init__(self, env):
        '''
        '''
        self.env = env
        self.state = None
        self.next_waypoint = None

    def reset(self, destination=None):
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

    def get_next_waypoint(self):
        '''
        '''
        return self.next_waypoint


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

    def update(self, t):
        '''
        '''
        inputs = self.env.sense(self)

        action_okay = True
        if self.next_waypoint == 'right':
            if inputs['light'] == 'red' and inputs['left'] == 'forward':
                action_okay = False
        elif self.next_waypoint == 'forward':
            if inputs['light'] == 'red':
                action_okay = False
        elif self.next_waypoint == 'left':
            if inputs['light'] == 'red' or (inputs['oncoming'] == 'forward' or inputs['oncoming'] == 'right'):
                action_okay = False

        action = None
        if action_okay:
            action = self.next_waypoint
            self.next_waypoint = random.choice(Environment.valid_actions[1:])
        reward = self.env.act(self, action)
