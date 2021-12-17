"""
The velociraptor problem, aka xkcd 135, thanks to Randall Munroe.
"""

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
from gym.envs.registration import register
import numpy as np
from scipy.constants import tau

class Player():

    def __init__(self):
        self.pos = (0,0)
        self.v = 6.0

    def move(action, delta_t):
        theta = action[0]
        self.pos = (self.pos[0] + 6. * delta_t * np.cos(theta),
                    self.pos[1] + 6. * delta_t * np.sin(theta))

class Raptor():

    def __init__(self, pos, injured = False):
        self.pos = pos
        self.v = 0.0
        self.v_max = (25. if not injured else 10.)

    def move(player):
        self.v = np.clip(0., self.v + 4., self.v_max)
        delta_x = player.pos[0] - self.pos[0]
        delta_y = player.pos[1] - self.pos[1]
        r = np.sqrt(delta_x**2 + delta_y**2)
        theta = np.arctan(delta_y / delta_x)
        self.pos = (self.pos[0] + max(self.v * delta_t * np.cos(theta), r),
                    self.pos[1] + max(self.v * delta_t * np.sin(theta), r))

class VelociraptorEnv(gym.Env):
    """
    Description:
        You are at the centre of a 20 m equilateral triangle with a raptor at each corner.
        The top raptor has a wounded leg and is limited to a top speed of 10 ms-1.
        The other two have a top speed of 25 ms-1. All accelerate from 0 at 4 ms-2.
        You can run at 6 ms-1 immediately at any angle you choose.
        At what angle do you run to maximise the time you stay alive?

    Source:
        with apologies to Randall Munroe:
        https://xkcd.com/135

    Observation:
        3 raptor positions relative to yourself.

    Action:
        Angle to run at

    Reward:
        +1 for every millisecond spent alive

    Starting State:
        Player: (0,0)
        Raptors: ()

    Episode Termination:
        You have been eaten.
    """

    def __init__(self):
        self.player = Player()
        self.raptors = [Raptor((-10.,-5.77)), Raptor((-10.,5.77)), Raptor((0,11.),injured=True)]
        self.delta_t = 0.01

        self.action_space = spaces.Box(np.array([0]), np.array([tau]))

        self.viewer = None

    def step(self, action):
        assert self.action_space.contains(action), err_msg

        self.player.move(action, delta_t)
        for r in self.raptors:
            r.move(self.player.pos, delta_t)

        if any(r.pos == self.player.pos for r in self.raptors):
            done = True

        if not done:
            reward = 1.0

        return np.array(self.state, dtype=np.float32), reward, done, {}

    def reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        return np.array(self.state, dtype=np.float32)

    def render(self):
        #TODO: add actual files and check how gym rendering works
        if self.viewer is None:
            self.viewer = rendering.Viewer(800,800)

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

register(
        id="velociraptor-v1",
        entry_point="__main__:VelociraptorEnv",
        max_episode_steps=20000,
        reward_threshold=195.0
    )

env = gym.make("velociraptor-v1")
env.reset()

#TODO: example agent with stable_baselines
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample()) # take a random action
env.close()
