import embodied
import numpy as np

class Snake(embodied.Env):

  def __init__(self, task):
    from . import from_gym
    assert task in ('reward', 'noreward')
    import gym_snake_game
    from gym_snake_game import environment
    self._gymenv = environment.SnakeEnv(width=64,height=64,block_size=40)
    self._env = from_gym.FromGym(self._gymenv)

  @property
  def obs_space(self):
    return self._env.obs_space

  @property
  def act_space(self):
    return self._env.act_space

  def step(self, action):
    obs = self._env.step(action)
    obs['is_terminal'] = False
    return obs