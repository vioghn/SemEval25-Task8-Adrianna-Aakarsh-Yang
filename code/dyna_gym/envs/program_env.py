# Yang: a new env for code generating

from collections import OrderedDict
import gymnasium as gym
import torch
import transformers
from transformers import pipeline
from gymnasium.spaces import Discrete, Box
import numpy as np


class ProgramEnv(gym.Env):
    """
    Code generation environment.

    State: a list of tokens.
    Action: a token (an integer).
    Reward: pass rate of the program (on the training set in training, and on the test set in testing).
    """

    def __init__(self, terminal_token, horizon, reward_func, vocab_size):
        """
        Args:
            terminal_token: The token for the terminal action
            horizon: the maximum length including the prompt
        """
        self.terminal_token = terminal_token
        self.horizon = horizon
        self.get_reward = reward_func
        self.vocab_size = vocab_size
        self.action_space = Discrete(vocab_size)
        self.observation_space = Box(low=0, high=vocab_size, shape=(horizon,))

    def reset(self, input_ids, attention_mask=None):
        # ini or reset the environment to give a starting state
        if attention_mask is not None:
            attention_mask = attention_mask
        else:
            attention_mask = torch.ones_like(input_ids)

        self.state = (input_ids, attention_mask)
        self.input_len = len(input_ids)
        return self.state

    def transition(self, s, a, is_model_dynamic=True):
        """
        s: current state, which is a tuple (ids, attention_mask)
        a: action, a token ID
        is_model_dynamic: placeholder
        """
        # ids, attention_mask = s
        # s is a one-dimensional tensor, a is a token id (scalar), concatenate them to form a new state
        # next_ids = torch.cat([ids, torch.tensor([a]).to(ids.device)])
        # append a 1 to the attention mask
        # attention_mask = torch.cat([attention_mask, torch.tensor([1]).to(attention_mask.device)])

        next_state = s + [a]
        if a == self.terminal_token or len(next_state) == self.horizon:
            # either the program finishes, or the state reaches the maximum length
            done = True
        else:
            done = False
        if done:
            reward = self.get_reward(next_state)
        else:
            reward = 0  # no intermediate reward
        return next_state, reward, done

    def step(self, action):
        self.state, reward, done = self.transition(self.state, action)
        return self.state, reward, done, {}

    def equality_operator(self, s1, s2):
        return s1 == s2



