
import math
from collections import deque
from time import perf_counter

import gymnasium as gym
import numpy as np
from gymnasium import ObservationWrapper, spaces
from gymnasium.wrappers import TimeLimit as GymTimeLimit
from gymnasium.wrappers import RecordVideo as GymMonitor

class RecordEpisodeStatistics(gym.Wrapper):
    """ Multi-agent version of RecordEpisodeStatistics gym wrapper"""

    def __init__(self, env, deque_size=100):
        super().__init__(env)
        self.env = env
        self.t0 = perf_counter()
        self.episode_reward = np.zeros(self._get_n_agents(env))
        self.episode_length = 0
        self.reward_queue = deque(maxlen=deque_size)
        self.length_queue = deque(maxlen=deque_size)

    def _get_n_agents(self, env):
        while hasattr(env, 'env'):
            if hasattr(env, 'n_agents'):
                return env.n_agents
            env = env.env
        return env.n_agents

    def reset(self, **kwargs):
        observation = super().reset(**kwargs)
        self.episode_reward = np.zeros(self._get_n_agents(self.env))
        self.episode_length = 0
        self.t0 = perf_counter()

        return observation

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        print("OB:", len(observation))
        done = [terminated, truncated]

            
        reward = sum(reward.values()) if isinstance(reward, dict) else reward
        
        self.episode_reward += np.array(reward, dtype=np.float64)
        print("REWARD:",self.episode_reward)
        self.episode_length += 1
        if all(done):
            info["episode_reward"] = self.episode_reward
            for i, agent_reward in enumerate(self.episode_reward):
                info[f"agent{i}/episode_reward"] = agent_reward
            info["episode_length"] = self.episode_length
            info["episode_time"] = perf_counter() - self.t0

            self.reward_queue.append(self.episode_reward)
            self.length_queue.append(self.episode_length)
            
        

        return observation, reward, done, info


class FlattenObservation(ObservationWrapper):
    r"""Observation wrapper that flattens the observation of individual agents."""

    def __init__(self, env):
        super(FlattenObservation, self).__init__(env)

        ma_spaces = []

        for sa_obs in env.observation_space:
            flatdim = spaces.flatdim(sa_obs)
            ma_spaces += [
                spaces.Box(
                    low=-float("inf"),
                    high=float("inf"),
                    shape=(flatdim,),
                    dtype=np.float32,
                )
            ]

        self.observation_space = spaces.Tuple(tuple(ma_spaces))

    def observation(self, observation):
        return tuple([
            spaces.flatten(obs_space, obs)
            for obs_space, obs in zip(self.env.observation_space, observation)
        ])


class SquashDones(gym.Wrapper):
    r"""Wrapper that squashes multiple dones to a single one using all(dones)"""

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        # done = all(terminated) or all(truncated)
        return observation[0], reward, all(done), info


class GlobalizeReward(gym.RewardWrapper):
    def reward(self, reward):
        return self.n_agents * [sum(reward)]


class TimeLimit(GymTimeLimit):
    def __init__(self, env, max_episode_steps=None):
        super().__init__(env)
        if max_episode_steps is None and self.env.spec is not None:
            max_episode_steps = env.spec.max_episode_steps
        # if self.env.spec is not None:
        #     self.env.spec.max_episode_steps = max_episode_steps
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None

    def step(self, action):
        assert self._elapsed_steps is not None, "Cannot call env.step() before calling reset()"
        observation, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            info['TimeLimit.truncated'] = not all(done)
            done = len(observation) * [True]
        return observation, reward, done, info

class ClearInfo(gym.Wrapper):
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return observation, reward, done, {}


class Monitor(GymMonitor):
    def _after_step(self, observation, reward, done, info):
        if not self.enabled: return done

        if all(done) and self.env_semantics_autoreset:
            # For envs with BlockingReset wrapping VNCEnv, this observation will be the first one of the new episode
            self.reset_video_recorder()
            self.episode_id += 1
            self._flush()

        # Record stats
        self.stats_recorder.after_step(observation, sum(reward), all(done), info)
        # Record video
        self.video_recorder.capture_frame()

        return done
