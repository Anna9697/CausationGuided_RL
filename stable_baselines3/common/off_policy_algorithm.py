import io
import pathlib
import math
import pickle
import time
import warnings
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gym
import numpy as np
import torch as th
import matplotlib.pyplot as plt

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.buffers import DictReplayBuffer, ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.save_util import load_from_pkl, save_to_pkl
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, RolloutReturn, Schedule, TrainFreq, TrainFrequencyUnit
from stable_baselines3.common.utils import safe_mean, should_collect_more_steps
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
import sys
import stl_robustness
import stl_robustness_lse
import stl_robustness_sss
import stl_causation_opt
import stl_causation_app
from collections import deque
import gc
import objgraph
import csv
import psutil, os, time


#fulllog=[]

class OffPolicyAlgorithm(BaseAlgorithm):
    """
    The base for Off-Policy algorithms (ex: SAC/TD3)

    :param policy: Policy object
    :param env: The environment to learn from
                (if registered in Gym, can be str. Can be None for loading trained models)
    :param policy_base: The base policy used by this method
    :param learning_rate: learning rate for the optimizer,
        it can be a function of the current progress remaining (from 1 to 0)
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1)
    :param gamma: the discount factor
    :param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
        like ``(5, "step")`` or ``(2, "episode")``.
    :param gradient_steps: How many gradient steps to do after each rollout (see ``train_freq``)
        Set to ``-1`` means to do as many gradient steps as steps done in the environment
        during the rollout.
    :param action_noise: the action noise type (None by default), this can help
        for hard exploration problem. Cf common.noise for the different action noise type.
    :param replay_buffer_class: Replay buffer class to use (for instance ``HerReplayBuffer``).
        If ``None``, it will be automatically selected.
    :param replay_buffer_kwargs: Keyword arguments to pass to the replay buffer on creation.
    :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
    :param policy_kwargs: Additional arguments to be passed to the policy on creation
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param verbose: The verbosity level: 0 none, 1 training information, 2 debug
    :param device: Device on which the code should run.
        By default, it will try to use a Cuda compatible device and fallback to cpu
        if it is not possible.
    :param support_multi_env: Whether the algorithm supports training
        with multiple environments (as in A2C)
    :param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param monitor_wrapper: When creating an environment, whether to wrap it
        or not in a Monitor wrapper.
    :param seed: Seed for the pseudo random generators
    :param use_sde: Whether to use State Dependent Exploration (SDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param use_sde_at_warmup: Whether to use gSDE instead of uniform sampling
        during the warm up phase (before learning starts)
    :param sde_support: Whether the model support gSDE or not
    :param remove_time_limit_termination: Remove terminations (dones) that are due to time limit.
        See https://github.com/hill-a/stable-baselines/issues/863
    :param supported_action_spaces: The action spaces supported by the algorithm.
    """

    def __init__(
        self,
        policy: Type[BasePolicy],
        env: Union[GymEnv, str],
        policy_base: Type[BasePolicy],
        learning_rate: Union[float, Schedule],
        buffer_size: int = 1000000,  # 1e6
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = (1, "step"),
        gradient_steps: int = 1,
        action_noise: Optional[ActionNoise] = None,
        replay_buffer_class: Optional[ReplayBuffer] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        tensorboard_log: Optional[str] = None,
        verbose: int = 0,
        device: Union[th.device, str] = "auto",
        support_multi_env: bool = False,
        create_eval_env: bool = False,
        monitor_wrapper: bool = True,
        seed: Optional[int] = None,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        use_sde_at_warmup: bool = False,
        sde_support: bool = True,
        remove_time_limit_termination: bool = False,
        supported_action_spaces: Optional[Tuple[gym.spaces.Space, ...]] = None,
    ):

        super(OffPolicyAlgorithm, self).__init__(
            policy=policy,
            env=env,
            policy_base=policy_base,
            learning_rate=learning_rate,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            support_multi_env=support_multi_env,
            create_eval_env=create_eval_env,
            monitor_wrapper=monitor_wrapper,
            seed=seed,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            supported_action_spaces=supported_action_spaces,
        )
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.tau = tau
        self.gamma = gamma
        self.gradient_steps = gradient_steps
        self.action_noise = action_noise
        self.optimize_memory_usage = optimize_memory_usage
        self.replay_buffer_class = replay_buffer_class
        if replay_buffer_kwargs is None:
            replay_buffer_kwargs = {}
        self.replay_buffer_kwargs = replay_buffer_kwargs
        self._episode_storage = None

        # Remove terminations (dones) that are due to time limit
        # see https://github.com/hill-a/stable-baselines/issues/863
        self.remove_time_limit_termination = remove_time_limit_termination

        # Save train freq parameter, will be converted later to TrainFreq object
        self.train_freq = train_freq

        self.actor = None  # type: Optional[th.nn.Module]
        self.replay_buffer = None  # type: Optional[ReplayBuffer]
        # Update policy keyword arguments
        if sde_support:
            self.policy_kwargs["use_sde"] = self.use_sde
        # For gSDE only
        self.use_sde_at_warmup = use_sde_at_warmup
        self._last_obs = None
        self._last_original_obs = None

    def _convert_train_freq(self) -> None:
        """
        Convert `train_freq` parameter (int or tuple)
        to a TrainFreq object.
        """
        if not isinstance(self.train_freq, TrainFreq):
            train_freq = self.train_freq

            # The value of the train frequency will be checked later
            if not isinstance(train_freq, tuple):
                train_freq = (train_freq, "step")

            try:
                train_freq = (train_freq[0], TrainFrequencyUnit(train_freq[1]))
            except ValueError:
                raise ValueError(f"The unit of the `train_freq` must be either 'step' or 'episode' not '{train_freq[1]}'!")

            if not isinstance(train_freq[0], int):
                raise ValueError(f"The frequency of `train_freq` must be an integer and not {train_freq[0]}")

            self.train_freq = TrainFreq(*train_freq)

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        # Use DictReplayBuffer if needed
        if self.replay_buffer_class is None:
            if isinstance(self.observation_space, gym.spaces.Dict):
                self.replay_buffer_class = DictReplayBuffer
            else:
                self.replay_buffer_class = ReplayBuffer

        elif self.replay_buffer_class == HerReplayBuffer:
            assert self.env is not None, "You must pass an environment when using `HerReplayBuffer`"

            # If using offline sampling, we need a classic replay buffer too
            if self.replay_buffer_kwargs.get("online_sampling", True):
                replay_buffer = None
            else:
                replay_buffer = DictReplayBuffer(
                    self.buffer_size,
                    self.observation_space,
                    self.action_space,
                    self.device,
                    optimize_memory_usage=self.optimize_memory_usage,
                )

            self.replay_buffer = HerReplayBuffer(
                self.env,
                self.buffer_size,
                self.device,
                replay_buffer=replay_buffer,
                **self.replay_buffer_kwargs,
            )

        if self.replay_buffer is None:
            self.replay_buffer = self.replay_buffer_class(
                self.buffer_size,
                self.observation_space,
                self.action_space,
                self.device,
                optimize_memory_usage=self.optimize_memory_usage,
                **self.replay_buffer_kwargs,
            )

        self.policy = self.policy_class(  # pytype:disable=not-instantiable
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            **self.policy_kwargs,  # pytype:disable=not-instantiable
        )
        self.policy = self.policy.to(self.device)

        # Convert train freq parameter to TrainFreq object
        self._convert_train_freq()

    def save_replay_buffer(self, path: Union[str, pathlib.Path, io.BufferedIOBase]) -> None:
        """
        Save the replay buffer as a pickle file.

        :param path: Path to the file where the replay buffer should be saved.
            if path is a str or pathlib.Path, the path is automatically created if necessary.
        """
        assert self.replay_buffer is not None, "The replay buffer is not defined"
        save_to_pkl(path, self.replay_buffer, self.verbose)

    def load_replay_buffer(
        self,
        path: Union[str, pathlib.Path, io.BufferedIOBase],
        truncate_last_traj: bool = True,
    ) -> None:
        """
        Load a replay buffer from a pickle file.

        :param path: Path to the pickled replay buffer.
        :param truncate_last_traj: When using ``HerReplayBuffer`` with online sampling:
            If set to ``True``, we assume that the last trajectory in the replay buffer was finished
            (and truncate it).
            If set to ``False``, we assume that we continue the same trajectory (same episode).
        """
        self.replay_buffer = load_from_pkl(path, self.verbose)
        assert isinstance(self.replay_buffer, ReplayBuffer), "The replay buffer must inherit from ReplayBuffer class"

        # Backward compatibility with SB3 < 2.1.0 replay buffer
        # Keep old behavior: do not handle timeout termination separately
        if not hasattr(self.replay_buffer, "handle_timeout_termination"):  # pragma: no cover
            self.replay_buffer.handle_timeout_termination = False
            self.replay_buffer.timeouts = np.zeros_like(self.replay_buffer.dones)

        if isinstance(self.replay_buffer, HerReplayBuffer):
            assert self.env is not None, "You must pass an environment at load time when using `HerReplayBuffer`"
            self.replay_buffer.set_env(self.get_env())
            if truncate_last_traj:
                self.replay_buffer.truncate_last_trajectory()

    def _setup_learn(
        self,
        total_timesteps: int,
        eval_env: Optional[GymEnv],
        callback: MaybeCallback = None,
        eval_freq: int = 10000,
        n_eval_episodes: int = 5,
        log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
        tb_log_name: str = "run",
    ) -> Tuple[int, BaseCallback]:
        """
        cf `BaseAlgorithm`.
        """
        # Prevent continuity issue by truncating trajectory
        # when using memory efficient replay buffer
        # see https://github.com/DLR-RM/stable-baselines3/issues/46

        # Special case when using HerReplayBuffer,
        # the classic replay buffer is inside it when using offline sampling
        if isinstance(self.replay_buffer, HerReplayBuffer):
            replay_buffer = self.replay_buffer.replay_buffer
        else:
            replay_buffer = self.replay_buffer

        truncate_last_traj = (
            self.optimize_memory_usage
            and reset_num_timesteps
            and replay_buffer is not None
            and (replay_buffer.full or replay_buffer.pos > 0)
        )

        if truncate_last_traj:
            warnings.warn(
                "The last trajectory in the replay buffer will be truncated, "
                "see https://github.com/DLR-RM/stable-baselines3/issues/46."
                "You should use `reset_num_timesteps=False` or `optimize_memory_usage=False`"
                "to avoid that issue."
            )
            # Go to the previous index
            pos = (replay_buffer.pos - 1) % replay_buffer.buffer_size
            replay_buffer.dones[pos] = True

        return super()._setup_learn(
            total_timesteps,
            eval_env,
            callback,
            eval_freq,
            n_eval_episodes,
            log_path,
            reset_num_timesteps,
            tb_log_name,
        )

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "run",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
        reward_type: str = "nominal",
        stl: str = "stl",
        sem: str = "cau",
        run: str = "test",
        data_len: int = 1,
        max_episodes: int = 1e4,
    ) -> "OffPolicyAlgorithm":

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            eval_env,
            callback,
            eval_freq,
            n_eval_episodes,
            eval_log_path,
            reset_num_timesteps,
            tb_log_name,
        )

        callback.on_training_start(locals(), globals())

        data_in = deque(maxlen=data_len)
        last_ep = 0;

        reward_list, reward_avg_list, step_list, reward_perstep = [0], [], [0], []
        satisfy = [True]
        pre_rob = [0.0]

        while self.num_timesteps < total_timesteps:

            rollout = self.collect_rollouts(
                data_in,
                reward_list,
                reward_avg_list,
                step_list,
                reward_perstep,
                satisfy,
                pre_rob,
                self.env,
                train_freq=self.train_freq,
                action_noise=self.action_noise,
                callback=callback,
                learning_starts=self.learning_starts,
                replay_buffer=self.replay_buffer,
                log_interval=log_interval,
                reward_type=reward_type,
                sem=sem,
                stl=stl,
            )

            if rollout.continue_training is False:
                break

            if self.num_timesteps > 0 and self.num_timesteps > self.learning_starts:
                # If no `gradient_steps` is specified,
                # do as many gradients steps as steps performed during the rollout
                gradient_steps = self.gradient_steps if self.gradient_steps >= 0 else rollout.episode_timesteps
                # Special case when the user passes `gradient_steps=0`
                if gradient_steps > 0:
                    self.train(batch_size=self.batch_size, gradient_steps=gradient_steps)
                    # gc.collect()
                    # th.cuda.empty_cache()
                # print(rollout.reward_list)
                # print(rollout.reward_avg_list)
            # print(rollout.n_episodes)
            # if self._episode_num >10 and self._episode_num-last_ep>0:
            #     last_ep = self._episode_num
            #     self.plot_res(reward_list, reward_avg_list, step_list, fign='dqn_agent_train.png',save_fig=False)
            # print(self._episode_num)
            if self._episode_num >= max_episodes:
                break


        # callback.on_training_end()

        env_name = self.env.envs[0].unwrapped.spec.id
        fign = "./result/fig/" + env_name + "_" + stl + "_" + sem + "_" + run + ".png"
        reward_file_name = "./result/list/" + env_name + "_" + "reward_" + stl + "_" + sem + "_" + run + ".pkl"
        reward_avg_file_name = "./result/list/" + env_name + "_" + "reward_avg_" + stl + "_" + sem + "_" + run + ".pkl"
        reward_perstep_file_name = "./result/list/" + env_name + "_reward_perstep_" + stl + "_" + sem + "_" + run + ".pkl"
        with open(reward_file_name, 'wb') as reward_file:
            pickle.dump(reward_list, reward_file)
        with open(reward_avg_file_name, 'wb') as reward_avg_file:
            pickle.dump(reward_avg_list, reward_avg_file)
        with open(reward_perstep_file_name, 'wb') as reward_perstep_file:
            pickle.dump(reward_perstep, reward_perstep_file)
        # print(total_timesteps, reward_perstep)
        self.plot_res(reward_list, reward_avg_list, step_list, reward_perstep, total_timesteps, fign, save_fig=True)
        print("Fig saved in ", fign)

        # self.plot_step(reward_perstep, total_timesteps, fign='dqn_agent_train.png', save_fig=True)

        del data_in
        # gc.collect()

        return self

    def plot_res(self, values, avg_values, steps, reward_perstep, total_timesteps, fign, save_fig=True):
        ''' Plot the reward curve and histogram of results over time.'''
        # Update the window after each episode
        # clear_output(wait=True)
        avg_steps = []
        for i in range(len(steps)):
            if i < 9:
                avg_steps.append(None)
            else:
                avg_steps.append(np.mean(steps[i - 9 : i+1]))

        # Define the figure
        f, ax = plt.subplots(nrows=1, ncols=3, figsize=(20, 5))
        f.suptitle('Training DQN Causation')
        ax[0].plot(values[1:], label='score per run')
        # ax[0].axhline(450, c='red', ls='--', label='goal')
        ax[0].plot(avg_values, label='average score per 50 run')
        ax[0].set_xlabel('Episodes')
        ax[0].set_ylabel('Reward')
        x = range(len(values))
        ax[0].legend()
        # Calculate the trend
        # try:
        #     z = np.polyfit(x, values, 1)
        #     p = np.poly1d(z)
        #     ax[0].plot(x, p(x), "--", label='trend')
        # except:
        #     print('')

        ax[1].plot(steps, label='steps per run')
        # ax[1].axhline(450, c='red', ls='--', label='goal')
        ax[1].plot(avg_steps, label='average step per 50 run')
        ax[1].set_xlabel('Episodes')
        ax[1].set_ylabel('Steps')
        x = range(len(steps))
        ax[1].legend()

        ax[2].plot(reward_perstep, label='reward per step')
        ax[2].set_xlabel('Steps')
        ax[2].set_ylabel('Reward')
        x = range(int(total_timesteps/100))
        ax[2].legend()

        if save_fig:
            plt.savefig(fign)

    def train(self, gradient_steps: int, batch_size: int) -> None:
        """
        Sample the replay buffer and do the updates
        (gradient descent and update target networks)
        """
        raise NotImplementedError()

    def _sample_action(
        self, learning_starts: int, action_noise: Optional[ActionNoise] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sample an action according to the exploration policy.
        This is either done by sampling the probability distribution of the policy,
        or sampling a random action (from a uniform distribution over the action space)
        or by adding noise to the deterministic output.

        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :return: action to take in the environment
            and scaled action that will be stored in the replay buffer.
            The two differs when the action space is not normalized (bounds are not [-1, 1]).
        """
        # Select action randomly or according to policy
        if self.num_timesteps < learning_starts and not (self.use_sde and self.use_sde_at_warmup):
            # Warmup phase
            unscaled_action = np.array([self.action_space.sample()])
        else:
            # Note: when using continuous actions,
            # we assume that the policy uses tanh to scale the action
            # We use non-deterministic action in the case of SAC, for TD3, it does not matter
            unscaled_action, _ = self.predict(self._last_obs, deterministic=False)

        # Rescale the action from [low, high] to [-1, 1]
        if isinstance(self.action_space, gym.spaces.Box):
            scaled_action = self.policy.scale_action(unscaled_action)

            # Add noise to the action (improve exploration)
            if action_noise is not None:
                scaled_action = np.clip(scaled_action + action_noise(), -1, 1)

            # We store the scaled action in the buffer
            buffer_action = scaled_action
            action = self.policy.unscale_action(scaled_action)
        else:
            # Discrete case, no need to normalize or clip
            buffer_action = unscaled_action
            action = buffer_action
        return action, buffer_action

    def _dump_logs(self) -> None:
        """
        Write log.
        """
        time_elapsed = time.time() - self.start_time
        fps = int(self.num_timesteps / (time_elapsed + 1e-8))
        self.logger.record("time/episodes", self._episode_num, exclude="tensorboard")
        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
            self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
            self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
        self.logger.record("time/fps", fps)
        self.logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
        self.logger.record("time/total timesteps", self.num_timesteps, exclude="tensorboard")
        if self.use_sde:
            self.logger.record("train/std", (self.actor.get_std()).mean().item())

        if len(self.ep_success_buffer) > 0:
            self.logger.record("rollout/success rate", safe_mean(self.ep_success_buffer))
        # Pass the number of timesteps for tensorboard
        self.logger.dump(step=self.num_timesteps)

    def _on_step(self) -> None:
        """
        Method called after each step in the environment.
        It is meant to trigger DQN target network update
        but can be used for other purposes
        """
        pass

    def _store_transition(
        self,
        replay_buffer: ReplayBuffer,
        buffer_action: np.ndarray,
        new_obs: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        """
        Store transition in the replay buffer.
        We store the normalized action and the unnormalized observation.
        It also handles terminal observations (because VecEnv resets automatically).

        :param replay_buffer: Replay buffer object where to store the transition.
        :param buffer_action: normalized action
        :param new_obs: next observation in the current episode
            or first observation of the episode (when done is True)
        :param reward: reward for the current transition
        :param done: Termination signal
        :param infos: List of additional information about the transition.
            It may contain the terminal observations and information about timeout.
        """
        # Store only the unnormalized version
        if self._vec_normalize_env is not None:
            new_obs_ = self._vec_normalize_env.get_original_obs()
            reward_ = self._vec_normalize_env.get_original_reward()
        else:
            # Avoid changing the original ones
            self._last_original_obs, new_obs_, reward_ = self._last_obs, new_obs, reward

        # As the VecEnv resets automatically, new_obs is already the
        # first observation of the next episode
        if done and infos[0].get("terminal_observation") is not None:
            next_obs = infos[0]["terminal_observation"]
            # VecNormalize normalizes the terminal observation
            if self._vec_normalize_env is not None:
                next_obs = self._vec_normalize_env.unnormalize_obs(next_obs)
        else:
            next_obs = new_obs_

        replay_buffer.add(
            self._last_original_obs,
            next_obs,
            buffer_action,
            reward_,
            done,
            infos,
        )

        self._last_obs = new_obs
        # Save the unnormalized observation
        if self._vec_normalize_env is not None:
            self._last_original_obs = new_obs_



    def collect_rollouts(
        self,
        data_in,
        reward_list,
        reward_avg_list,
        step_list,
        reward_perstep,
        satisfy,
        pre_rob,
        env: VecEnv,
        callback: BaseCallback,
        train_freq: TrainFreq,
        replay_buffer: ReplayBuffer,
        action_noise: Optional[ActionNoise] = None,
        learning_starts: int = 0,
        log_interval: Optional[int] = None,
        reward_type : str = "nominal",
        sem: str = "cau",
        stl : str = "stl",
    ) -> RolloutReturn:
        """
        Collect experiences and store them into a ``ReplayBuffer``.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param train_freq: How much experience to collect
            by doing rollouts of current policy.
            Either ``TrainFreq(<n>, TrainFrequencyUnit.STEP)``
            or ``TrainFreq(<n>, TrainFrequencyUnit.EPISODE)``
            with ``<n>`` being an integer greater than 0.
        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param replay_buffer:
        :param log_interval: Log data every ``log_interval`` episodes
        :return:
        """
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        # episode_rewards, reward_avg_list, total_timesteps = [], [], []
        episode_rewards, total_timesteps = [], []
        num_collected_steps, num_collected_episodes = 0, 0
        #ctr=0
        # lognnn = log_interval
        # print("--------------------------")



        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert env.num_envs == 1, "OffPolicyAlgorithm only support single environment"
        assert train_freq.frequency > 0, "Should at least collect one step or episode."

        if self.use_sde:
            self.actor.reset_noise()

        callback.on_rollout_start()
        continue_training = True
        global spec, stl_buf, signal_buf
        # satisfy = True
        # pre_rob = 0.0
        done = False

        # data_in = deque(maxlen=20)

        while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
        # while True:

            done = False
            # satisfy = True
            # pre_rob = 0.0
            episode_reward, episode_timesteps = 0.0, 0
            # episode_rewards[self._episode_num] = 0.0
            # print("+++++++++++++++++++++++++++")
            # i=0

            #ctr = 0

            # data buffer for causation
            # print(env.envs[0].dt)
            while not done:
                # i+=1

                if self.use_sde and self.sde_sample_freq > 0 and num_collected_steps % self.sde_sample_freq == 0:
                    # Sample a new noise matrix
                    self.actor.reset_noise()

                # Select action randomly or according to policy
                action, buffer_action = self._sample_action(learning_starts, action_noise)
                #print(action)
                #print(buffer_action)
                #exit()
                # Rescale and perform action
                new_obs, reward, done, infos = env.step(action)
                # print("done:", done)
                # env.render()
                #print(reward_type)
                name = env.envs[0].unwrapped.spec.id
                episode_reward += reward
                                                                    
                if name=="Hopper-v3" and reward_type=="STL":
                    #print(observation)
                    # = env.state # th := theta
                    #print(infos)
                    #exit()
                    # print(len(new_obs[0]))
                    obs = new_obs[0][-11:]
                    p = infos[0]['x_position']
                    v = infos[0]['x_velocity']
                    z, a = obs[0:2]
                    #print(str(h)+"  "+str(a))
                    u=action
                    if callback.ctr==0:
                        if stl == "stl":
                            signal_buf = 'p,v,z,a'
                            stl_buf = 'alw_[0,1000](ev_[0,15](v[t]>0.5) and (z[t]>0.7) and (a[t]<1))'
                    if stl=="stl":
                        data = np.array([callback.ctr, p, v, z, abs(a)])
                        data_in.append(data)
                        if sem == "cau":
                            robs = stl_causation_opt.mainFunction(signal_buf, stl_buf, np.array(data_in), np.array([0]))
                        elif sem == "cau_app":
                            robs = stl_causation_app.mainFunction(signal_buf, stl_buf, np.array(data_in), np.array([0]))
                        elif sem == "rob":
                            robs = stl_robustness.mainFunction(signal_buf, stl_buf, np.array(data_in), np.array([0]))
                        elif sem == "lse":
                            robs = stl_robustness_lse.mainFunction(signal_buf, stl_buf, np.array(data_in), np.array([0]))
                        elif sem == "sss":
                            robs = stl_robustness_sss.mainFunction(signal_buf, stl_buf, np.array(data_in), np.array([0]))

                        now_rob = robs[int(len(data_in)-1)]
                        if now_rob < 0:
                            satisfy[0] = False
                        rob = now_rob
                        if sem == "cau" or sem == "cau_app":
                            if now_rob > 0:
                                rob = 5.
                            if callback.ctr < 16:
                                rob = 1.0
                        if done[0] and callback.ctr >= 999 and satisfy[0]:
                            print("Satisfied!!!")
                        pre_rob[0] = now_rob
                    reward = rob

                elif name=="Walker2d-v3" and reward_type=="STL":
                    # print(len(new_obs[0]))
                    obs = new_obs[0][-17:]

                    px = infos[0]['x_position']
                    vx = infos[0]['x_velocity']
                    z , a = obs[0:2]
                    u=action
                    if callback.ctr==0:
                        if stl=="stl":
                            signal_buf = 'px,vx,z,a'
                            stl_buf = 'alw_[0,1500](ev_[0,15](vx[t]>0.5) and (z[t]<0.6) and (a[t]<1))'
                    if stl=="stl":
                        data = np.array([callback.ctr, px, vx, abs(z-1.4), abs(a)])
                        data_in.append(data)
                        if sem == "cau":
                            robs = stl_causation_opt.mainFunction(signal_buf, stl_buf, np.array(data_in), np.array([0]))
                        elif sem == "cau_app":
                            robs = stl_causation_app.mainFunction(signal_buf, stl_buf, np.array(data_in), np.array([0]))
                        elif sem == "rob":
                            robs = stl_robustness.mainFunction(signal_buf, stl_buf, np.array(data_in), np.array([0]))
                        elif sem == "lse":
                            robs = stl_robustness_lse.mainFunction(signal_buf, stl_buf, np.array(data_in), np.array([0]))
                        elif sem == "sss":
                            # print("??")
                            robs = stl_robustness_sss.mainFunction(signal_buf, stl_buf, np.array(data_in), np.array([0]))
                        now_rob = robs[int(len(data_in) - 1)]
                        if now_rob < 0:
                            satisfy[0] = False
                        rob = now_rob
                        if sem == "cau" or sem == "cau_app":
                            if now_rob > 0:
                                rob = 5
                            if callback.ctr < 16:
                                rob = 1.0
                        if done[0] and callback.ctr >= 999 and satisfy[0]:
                            print("Satisfied!!!")
                        pre_rob[0] = now_rob
                    reward=rob

                #############################################################################
                
                self.num_timesteps += 1
                step_list[self._episode_num] += 1
                episode_timesteps += 1
                # print(reward)

                num_collected_steps += 1
                #ctr += 1 
                #print("at the end : "+str(ctr)+" "+str(self.num_timesteps))

                # Give access to local variables
                callback.update_locals(locals())
                # Only stop training if return value is False, not when it is None.
                if callback.on_step() is False:
                    return RolloutReturn(0.0, num_collected_steps, num_collected_episodes, continue_training=False)

                reward_list[self._episode_num] += reward
                # episode_reward += reward


                # print(num_collected_steps)
                if self.num_timesteps % 1000 == 0:
                    if self._episode_num > 50:
                        reward_perstep.append(np.mean(reward_list[-50:]))
                    else:
                        reward_perstep.append(np.mean(reward_list))
                    # reward_perstep.append(safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))

                # Retrieve reward and episode length if using Monitor wrapper
                self._update_info_buffer(infos, done)

                # Store data in replay buffer (normalized action and unnormalized observation)
                self._store_transition(replay_buffer, buffer_action, new_obs, reward, done, infos)
                # print("stored obs id:", id(new_obs), "first buffer obs id:", id(env.envs[0].obs_buffer[0]))

                self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)

                # For DQN, check if the target network should be updated
                # and update the exploration schedule
                # For SAC/TD3, the update is done as the same time as the gradient update
                # see https://github.com/hill-a/stable-baselines/issues/900
                self._on_step()

                # print("should_collect_more_steps:", should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes))

                if not should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
                    break



            if done:
                print(reward_list[self._episode_num], step_list[self._episode_num])

                num_collected_episodes += 1
                self._episode_num += 1
                reward_list.append(0)
                step_list.append(0)

                episode_rewards.append(episode_reward)
                total_timesteps.append(episode_timesteps)

                satisfy[0] = True
                pre_rob[0] = 0.0

                # print(self._episode_num, episode_reward, episode_timesteps)

                if self._episode_num > 50:
                    avg_reward = np.mean(reward_list[-50:])
                else:
                    avg_reward = np.mean(reward_list)
                reward_avg_list.append(avg_reward)
                #print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>."+str(episode_timesteps))
                callback.ctr = 0

                if action_noise is not None:
                    action_noise.reset()

                # Log training infos
                if log_interval is not None and self._episode_num % log_interval == 0:
                    self._dump_logs()
                gc.collect()
                th.cuda.empty_cache()
                data_in.clear()
                spec = None
                signal_buf = None
                stl_buf = None


        mean_reward = np.mean(episode_rewards) if num_collected_episodes > 0 else 0.0

        callback.on_rollout_end()
        # gc.collect()

        return RolloutReturn(mean_reward, num_collected_steps, num_collected_episodes, continue_training)
