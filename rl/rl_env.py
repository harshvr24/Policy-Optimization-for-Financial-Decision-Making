#!/usr/bin/env python3
"""
rl/rl_env.py

A minimal single-step RL environment wrapper for evaluation and simulation.
Used to evaluate policy decisions on applicant rows.
"""

from __future__ import annotations
from typing import Sequence
import numpy as np
import gym
from gym import spaces

class SingleStepLoanEnv(gym.Env):
    """
    Single-step environment:
      - observation: continuous vector (features)
      - action: Discrete(2) -> 0 reject, 1 approve
      - reward: computed externally based on loan outcome (we pass reward per step)
    This class is useful for simulation: given an applicant, call step(action) to obtain reward.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, observation: np.ndarray):
        super().__init__()
        self.observation = observation.astype(np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=self.observation.shape, dtype=np.float32)
        self.action_space = spaces.Discrete(2)
        self._done = False

    def reset(self):
        self._done = False
        return self.observation

    def step(self, action: int, reward: float = 0.0):
        # returns (obs, reward, done, info)
        self._done = True
        info = {}
        return self.observation, float(reward), self._done, info

    def render(self, mode='human'):
        print("Single-step loan env")

    def close(self):
        pass
