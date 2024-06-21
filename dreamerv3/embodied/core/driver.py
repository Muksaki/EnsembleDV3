import collections

import numpy as np
import os

from .basics import convert


class Driver:
    _CONVERSION = {
        np.floating: np.float32,
        np.signedinteger: np.int32,
        np.uint8: np.uint8,
        bool: bool,
    }

    def __init__(self, env, **kwargs):
        assert len(env) > 0
        self._env = env
        self._kwargs = kwargs
        self._on_steps = []
        self._on_episodes = []
        self._success = []
        self.reset()

    def reset(self):
        self._acts = {
            k: convert(np.zeros((len(self._env),) + v.shape, v.dtype))
            for k, v in self._env.act_space.items()}
        self._acts['reset'] = np.ones(len(self._env), bool)
        self._eps = [collections.defaultdict(list) for _ in range(len(self._env))]
        self._state = None
        self._success = []

    def on_step(self, callback):
        self._on_steps.append(callback)

    def on_episode(self, callback):
        self._on_episodes.append(callback)

    def __call__(self, policy, steps=0, episodes=0, logger=None):
        step, episode = 0, 0
        while step < steps or episode < episodes:
            step, episode = self._step(policy, step, episode)
        if logger:
            self._success = np.array(self._success)
            logger.add({
                'success_rate': (np.sum(self._success) / episodes),
                }, prefix = 'eval_episode')
            # import ipdb; ipdb.set_trace()

    def _step(self, policy, step, episode):
        # import ipdb; ipdb.set_trace()
        assert all(len(x) == len(self._env) for x in self._acts.values())
        acts = {k: v for k, v in self._acts.items() if not k.startswith('log_')}
        obs = self._env.step(acts)
        obs = {k: convert(v) for k, v in obs.items()}
        assert all(len(x) == len(self._env) for x in obs.values()), obs
        import ipdb; ipdb.set_trace()
        acts, self._state = policy(obs, self._state, **self._kwargs)
        acts = {k: convert(v) for k, v in acts.items()}
        if obs['is_last'].any():
            mask = 1 - obs['is_last']
            acts = {k: v * self._expand(mask, len(v.shape)) for k, v in acts.items()}
        acts['reset'] = obs['is_last'].copy()
        self._acts = acts
        trns = {**obs, **acts}
        if obs['is_first'].any():
            for i, first in enumerate(obs['is_first']):
                if first:
                    self._eps[i].clear()
        for i in range(len(self._env)):
            trn = {k: v[i] for k, v in trns.items()}
            [self._eps[i][k].append(v) for k, v in trn.items()]
            [fn(trn, i, **self._kwargs) for fn in self._on_steps]
            step += 1
        if obs['is_last'].any():
            for i, done in enumerate(obs['is_last']):
                if done:
                    ep = {k: convert(v) for k, v in self._eps[i].items()}
                    self._success.append(float(np.sum(ep['log_success']) >= 1.0))
                    [fn(ep.copy(), i, **self._kwargs) for fn in self._on_episodes]
                    episode += 1
        return step, episode

    def _expand(self, value, dims):
        while len(value.shape) < dims:
            value = value[..., None]
        return value


class OfflineDriver:

  _CONVERSION = {
      np.floating: np.float32,
      np.signedinteger: np.int32,
      np.uint8: np.uint8,
      bool: bool,
  }

  

  ## TODO: Remove the dependency on specific envs
  def __init__(self, env, datadir, **kwargs):
    assert len(env) > 0
    self._env = env
    self._offline_datadir = datadir
    self._kwargs = kwargs
    self._on_steps = []
    self._on_episodes = []
    self.reset()
    self._load_keys = ('image', 'is_first', 'is_last', 'is_terminal', 'reward', 'action')
    self._obs_keys = ('image', 'is_first', 'is_last', 'is_terminal', 'reward')
    self._rollout_files = None
    self._rollout_step = None

  def reset(self):
    self._acts = {
        k: convert(np.zeros((len(self._env),) + v.shape, v.dtype))
        for k, v in self._env.act_space.items()}
    self._acts['reset'] = np.ones(len(self._env), bool)
    self._eps = [collections.defaultdict(list) for _ in range(len(self._env))]
    self._state = None

  def on_step(self, callback):
    self._on_steps.append(callback)

  def on_episode(self, callback):
    self._on_episodes.append(callback)

  def __call__(self):
    self._rollout_files = os.listdir(self._offline_datadir)
    for rollout_file in self._rollout_files:
      offline_ep = np.load(os.path.join(self._offline_datadir, rollout_file))
      for index in range(offline_ep['image'].shape[0]):
        tr = {k: [offline_ep[k][index]] for k in self._load_keys}
        self._step(tr)

  def _step(self, transitions):
    assert all(len(x) == len(self._env) for x in self._acts.values())
    acts = {k: v for k, v in self._acts.items() if not k.startswith('log_')}
    obs = {k: convert(v) for k, v in transitions.items() if k in self._obs_keys}
    # import ipdb; ipdb.set_trace()
    assert all(len(x) == len(self._env) for x in obs.values()), obs
    acts = {k: convert(v) for k, v in transitions.items() if not k in self._obs_keys}
    if obs['is_last'].any():
      mask = 1 - obs['is_last']
      acts = {k: v * self._expand(mask, len(v.shape)) for k, v in acts.items()}
    acts['reset'] = obs['is_last'].copy()
    self._acts = acts
    trns = {**obs, **acts}
    ## Warning: This is adjusted to be compatible with Town05_eps. Later will be changed to is_first
    if obs['is_first'].any():
      for i, first in enumerate(obs['is_first']):
        if first:
          self._eps[i].clear()
    for i in range(len(self._env)):
      trn = {k: v[i] for k, v in trns.items()}
      [self._eps[i][k].append(v) for k, v in trn.items()]
      [fn(trn, i, **self._kwargs) for fn in self._on_steps]
    if obs['is_last'].any():
      for i, done in enumerate(obs['is_last']):
        if done:
          ep = {k: convert(v) for k, v in self._eps[i].items()}
          [fn(ep.copy(), i, **self._kwargs) for fn in self._on_episodes]

  def _expand(self, value, dims):
    while len(value.shape) < dims:
      value = value[..., None]
    return value