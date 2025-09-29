import gym
import numpy as np
import copy
import gc
from collections import deque

class GymSeq(gym.ObservationWrapper):
    def __init__(self, env, n_steps=1):
        super(GymSeq, self).__init__(env)
        self.n_steps = n_steps
        self.obs_shape = env.observation_space.shape
        # 修改观察空间为 n * obs_shape
        self.observation_space = gym.spaces.Box(
            low=np.repeat(env.observation_space.low, self.n_steps, axis=0),
            high=np.repeat(env.observation_space.high, self.n_steps, axis=0),
            dtype=env.observation_space.dtype
        )
        # 状态缓冲区
        self.obs_buffer = deque(maxlen=self.n_steps)
        # self.obs_seq = None
        # self.info_buffer = deque(maxlen=self.n_steps)

    def reset(self):
        # 重置环境并初始化状态缓冲区
        obs = self.env.reset()
        for _ in range(self.n_steps):
            self.obs_buffer.append(obs)
        temp_buffer = copy.deepcopy(self.obs_buffer)
        obs_seq = np.concatenate(temp_buffer, axis=0)
        return obs_seq
        # return np.concatenate(temp_buffer, axis=0)

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        new_obs = self.observation(state)
        return new_obs, reward, done, info

    def observation(self, state):
        # 更新状态缓冲区并返回拼接后的观察
        self.obs_buffer.append(state)
        temp_buffer = copy.deepcopy(self.obs_buffer)
        obs_seq = np.concatenate(temp_buffer, axis=0)
        return obs_seq
        # return np.concatenate(temp_buffer, axis=0)