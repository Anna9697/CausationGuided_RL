import math
import pickle
import time
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import gym
import numpy as np
import torch as th
import matplotlib.pyplot as plt
from collections import deque

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy, BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import obs_as_tensor, safe_mean
from stable_baselines3.common.vec_env import VecEnv

import sys
import gc
#sys.path.insert(1, '/home/nikhil/RESEARCH/STL-RESEARCH/rtamt')
# import rtamt
import stl_robustness
import stl_causation_opt
import stl_causation_app
import stl_robustness_sss
import stl_robustness_lse

from highway_env.vehicle.controller import ControlledVehicle


class OnPolicyAlgorithm(BaseAlgorithm):
    """
    The base for On-Policy algorithms (ex: A2C/PPO).

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator.
        Equivalent to classic advantage when set to 1.
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param policy_base: The base policy used by this method
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param monitor_wrapper: When creating an environment, whether to wrap it
        or not in a Monitor wrapper.
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    :param supported_action_spaces: The action spaces supported by the algorithm.
    """

    def __init__(
        self,
        policy: Union[str, Type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule],
        n_steps: int,
        gamma: float,
        gae_lambda: float,
        ent_coef: float,
        vf_coef: float,
        max_grad_norm: float,
        use_sde: bool,
        sde_sample_freq: int,
        policy_base: Type[BasePolicy] = ActorCriticPolicy,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        create_eval_env: bool = False,
        monitor_wrapper: bool = True,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        supported_action_spaces: Optional[Tuple[gym.spaces.Space, ...]] = None,
    ):

        super(OnPolicyAlgorithm, self).__init__(
            policy=policy,
            env=env,
            policy_base=policy_base,
            learning_rate=learning_rate,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            create_eval_env=create_eval_env,
            support_multi_env=True,
            seed=seed,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            supported_action_spaces=supported_action_spaces,
        )

        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.rollout_buffer = None

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        buffer_cls = DictRolloutBuffer if isinstance(self.observation_space, gym.spaces.Dict) else RolloutBuffer
        # print("observation_space:", self.observation_space)

        self.rollout_buffer = buffer_cls(
            self.n_steps,
            self.observation_space,
            self.action_space,
            self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )
        self.policy = self.policy_class(  # pytype:disable=not-instantiable
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            use_sde=self.use_sde,
            **self.policy_kwargs  # pytype:disable=not-instantiable
        )
        self.policy = self.policy.to(self.device)


    def collect_rollouts(
        self,
        data_in,
        reward_list,
        reward_avg_list,
        step_list,
        satisfy,
        env: VecEnv,
        callback: BaseCallback,
        reward_type : str,
        mode : str,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
        sem: str = "cau",
        stl: str = "stl",
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        episode_rewards, total_timesteps = [], []
        episode_reward = 0.
        global spec, stl_buf, signal_buf, sat_buf

        n_steps = 0
        # i=0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()


        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                actions, values, log_probs = self.policy.forward(obs_tensor)
            actions = actions.cpu().numpy()
            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, gym.spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)
            # print("new_obs:",new_obs)
            # env.render()
            # print("Actions: ", clipped_actions, "\n New_obs: ", new_obs)
            # print("env_reward:", type(rewards[0]))
            name = self.env.envs[0].unwrapped.spec.id
            # print(reward_type)
            # print(reward_type)
            #exit()
            if name == "PointGoal1-v0" and reward_type == "STL":
                # print("obs_space_dict", env.envs[0].obs_space_dict)
                # print("obs_space_dict", sorted(env.envs[0].obs_space_dict.keys()))
                # print(new_obs)
                obs = new_obs[0][-44:]
                dg = max(obs[3:19])
                dc = max(obs[22:38])

                # Dg = (1 - dg) * env.envs[0].lidar_max_dist
                # Dc = (1 - dc) * env.envs[0].lidar_max_dist
                # print(Dg, Dc)

                u = clipped_actions
                if callback.ctr == 0:
                    if stl == "stl":
                        signal_buf = 'dg,dc'
                        stl_buf = 'alw_[0,1500](dc[t]<0.96667 and ev_[0,40](dg[t]>=0.95))'
                if stl == "stl":
                    # data = np.array([n_steps, Dg, Dc])
                    data = np.array([n_steps, dg, dc])
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
                    if n_steps >= 41:
                        rob = robs[int(len(data_in)) - 1]
                    else:
                        rob = 0.0
                    if rob < 0:
                        satisfy[0] = False
                        # rob = rob/20.0
                    if sem == "cau" or sem == "cau_app":
                        if rob > 0:
                            rob = 0.1
                    if dones[0] and n_steps >= 999 and satisfy[0]:
                        # rob += 10
                        print("Satisfied!!!")
                rewards = np.array([rob])

            if name == "CartPole-v1" and reward_type == "STL":
                obs = new_obs[0][-4:]
                loc = abs(obs[0])
                speed = abs(obs[1])
                angle = abs(obs[2])
                velocity = abs(obs[3])

                u = clipped_actions
                if callback.ctr == 0:
                    if stl == "stl":
                        signal_buf = 'loc,speed,angle,velocity'
                        stl_buf = 'alw_[0,1500](ev_[0,10](speed[t]<0.1) and loc[t]<0.5 and angle[t]<0.1)'
                if stl == "stl":
                    data = np.array([callback.ctr, loc, speed, angle, velocity])
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


                    if len(data_in)>=11:
                        rob = robs[int(len(data_in)) - 1]
                    else:
                        rob = 0
                    if rob < 0:
                        satisfy[0] = False
                    if sem == "cau" or sem == "cau_app":
                        if rob > 0:
                            rob = 1.0
                    if dones[0] and callback.ctr >= 499 and satisfy[0]:
                        # rob += 200
                        print("Satisfied!!!")
                rewards = np.array([rob])

            # print(rewards)
            self.num_timesteps += env.num_envs
            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False
            reward_list[-1] += rewards[0]
            step_list[-1] += 1
            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, gym.spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)
            rollout_buffer.add(self._last_obs, actions, rewards, self._last_episode_starts, values, log_probs)
            # self._last_obs = new_obs
            self._last_obs = new_obs
            self._last_episode_starts = dones

            # print(dones[0])
            # if n_steps==1 :
            #     print("??",reward_list[self._episode_num],callback.ctr)

            if dones[0]:
                reward_list.append(0)
                step_list.append(0)
                if self._episode_num > 50:
                    avg_reward = np.mean(reward_list[-50:])
                else:
                    avg_reward = np.mean(reward_list)
                reward_avg_list.append(avg_reward)
                print(reward_list[self._episode_num], callback.ctr)
                self._episode_num += 1
                # i = 0
                callback.ctr = 0
                # if name != "CartPole-v0":
                data_in.clear()

        with th.no_grad():
            # Compute value for the last timestep
            obs_tensor = obs_as_tensor(new_obs, self.device)
            # obs_tensor = obs_as_tensor(new_obs_seq, self.device)
            _, values, _ = self.policy.forward(obs_tensor)

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.on_rollout_end()

        return True

    def train(self) -> None:
        """
        Consume current rollout data and update policy parameters.
        Implemented by individual algorithms.
        """
        raise NotImplementedError

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "OnPolicyAlgorithm",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
        reward_type: str = "nominal",
        progress_bar: bool = False,
        mode: str = "train",
        stl: str = "stl",
        sem: str = "cau",
        run: str = "test",
        data_len: int = 1,
        max_episodes: int = 1e4,
    ) -> "OnPolicyAlgorithm":
        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps, eval_env, callback, eval_freq, n_eval_episodes, eval_log_path, reset_num_timesteps, tb_log_name, progress_bar,
        )

        callback.on_training_start(locals(), globals())

        if self.env.envs[0].unwrapped.spec.id == "PointWallGoal1-v0":
            data_in = deque(maxlen=data_len)
        else:
            data_in = deque(maxlen=data_len)
        reward_list, reward_avg_list, step_list = [0], [0], [0]
        satisfy = [True]
        # print("??")

        while self.num_timesteps < total_timesteps:

            continue_training = self.collect_rollouts(
                data_in,
                reward_list,
                reward_avg_list,
                step_list,
                satisfy,
                self.env,
                callback,
                reward_type,
                mode,
                self.rollout_buffer,
                n_rollout_steps=self.n_steps,
                sem=sem,
                stl=stl,
            )

            # if continue_training is False or (reward_avg_list[-1] and reward_avg_list[-1]>=500):
            if continue_training is False:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                fps = int(self.num_timesteps / (time.time() - self.start_time))
                self.logger.record("time/iterations", iteration, exclude="tensorboard")
                if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
                    self.logger.record("rollout/ep_rew_mean", safe_mean([ep_info["r"] for ep_info in self.ep_info_buffer]))
                    self.logger.record("rollout/ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer]))
                self.logger.record("time/fps", fps)
                self.logger.record("time/time_elapsed", int(time.time() - self.start_time), exclude="tensorboard")
                self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
                self.logger.dump(step=self.num_timesteps)

            if self.rollout_buffer.full:
                # print(True)
                self.train()
            # self.train()
            # print('----------')

            if self._episode_num >= max_episodes:
                break
        env_name = self.env.envs[0].unwrapped.spec.id
        fign = "./result/fig/" + env_name+ "_" + stl + "_" +sem + "_" + run + ".png"
        reward_file_name = "./result/list/" + env_name+ "_" + "reward_" + stl + "_" +sem + "_" + run + ".pkl"
        reward_avg_file_name = "./result/list/" + env_name + "_" + "reward_avg_" + stl + "_" + sem + "_" + run + ".pkl"
        with open(reward_file_name,'wb') as reward_file:
            pickle.dump(reward_list, reward_file)
        with open(reward_avg_file_name,'wb') as reward_avg_file:
            pickle.dump(reward_avg_list, reward_avg_file)
        self.plot_res(reward_list, reward_avg_list, step_list, fign, save_fig=True)
        print("Fig saved in ", fign)

        callback.on_training_end()

        return self

    def plot_res(self, values, avg_values, steps, fign, save_fig=True):
        ''' Plot the reward curve and histogram of results over time.'''
        # Update the window after each episode
        # clear_output(wait=True)

        # Define the figure
        f, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
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
        # ax[1].plot(avg_values, label='average score per 50 run')
        ax[1].set_xlabel('Episodes')
        ax[1].set_ylabel('Steps')
        x = range(len(steps))
        ax[1].legend()

        if save_fig:
            plt.savefig(fign)
        # else:
        # plt.show()

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "policy.optimizer"]

        return state_dicts, []
