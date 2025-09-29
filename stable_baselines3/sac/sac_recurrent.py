from copy import deepcopy
from typing import Any, ClassVar, Dict, List, Optional, Type, TypeVar, Union, Tuple

import gym
import numpy as np
import torch as th
from gym import spaces
from stable_baselines3.common.buffers import RolloutBuffer, RecurrentDictRolloutBuffer, RecurrentRolloutBuffer, ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
# from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy, LSTMPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule, RNNStates
from stable_baselines3.common.utils import explained_variance, get_schedule_fn, obs_as_tensor, polyak_update
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.noise import ActionNoise

import stl_robustness
import stl_causation_opt
import stl_causation_app
import stl_robustness_sss
import stl_robustness_lse

# from sb3_contrib.common.recurrent.buffers import RecurrentDictRolloutBuffer, RecurrentRolloutBuffer
# from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
# from sb3_contrib.common.recurrent.type_aliases import RNNStates
# from sb3_contrib.ppo_recurrent.policies import CnnLstmPolicy, MlpLstmPolicy, MultiInputLstmPolicy

SelfRecurrentPPO = TypeVar("SelfRecurrentPPO", bound="RecurrentPPO")

class RecurrentSAC(OffPolicyAlgorithm):
    """
    Soft Actor-Critic (SAC)
    Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor,
    This implementation borrows code from original implementation (https://github.com/haarnoja/sac)
    from OpenAI Spinning Up (https://github.com/openai/spinningup), from the softlearning repo
    (https://github.com/rail-berkeley/softlearning/)
    and from Stable Baselines (https://github.com/hill-a/stable-baselines)
    Paper: https://arxiv.org/abs/1801.01290
    Introduction to SAC: https://spinningup.openai.com/en/latest/algorithms/sac.html

    Note: we use double q target and not value target as discussed
    in https://github.com/hill-a/stable-baselines/issues/270

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: learning rate for adam optimizer,
        the same learning rate will be used for all networks (Q-Values, Actor and Value function)
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
    :param ent_coef: Entropy regularization coefficient. (Equivalent to
        inverse of reward scale in the original SAC paper.)  Controlling exploration/exploitation trade-off.
        Set it to 'auto' to learn it automatically (and 'auto_0.1' for using 0.1 as initial value)
    :param target_update_interval: update the target network every ``target_network_update_freq``
        gradient steps.
    :param target_entropy: target entropy when learning ``ent_coef`` (``ent_coef = 'auto'``)
    :param use_sde: Whether to use generalized State Dependent Exploration (gSDE)
        instead of action noise exploration (default: False)
    :param sde_sample_freq: Sample a new noise matrix every n steps when using gSDE
        Default: -1 (only sample at the beginning of the rollout)
    :param use_sde_at_warmup: Whether to use gSDE instead of uniform sampling
        during the warm up phase (before learning starts)
    :param create_eval_env: Whether to create a second environment that will be
        used for evaluating the agent periodically. (Only available when passing string for the environment)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: the verbosity level: 0 no output, 1 info, 2 debug
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    def __init__(
            self,
            policy: Union[str, Type[LSTMPolicy]],
            env: Union[GymEnv, str],
            learning_rate: Union[float, Schedule] = 3e-4,
            buffer_size: int = 1000000,  # 1e6
            learning_starts: int = 100,
            batch_size: int = 256,
            tau: float = 0.005,
            gamma: float = 0.99,
            train_freq: Union[int, Tuple[int, str]] = 1,
            gradient_steps: int = 1,
            action_noise: Optional[ActionNoise] = None,
            replay_buffer_class: Optional[ReplayBuffer] = None,
            replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
            optimize_memory_usage: bool = False,
            ent_coef: Union[str, float] = "auto",
            target_update_interval: int = 1,
            target_entropy: Union[str, float] = "auto",
            use_sde: bool = False,
            sde_sample_freq: int = -1,
            use_sde_at_warmup: bool = False,
            tensorboard_log: Optional[str] = None,
            create_eval_env: bool = False,
            policy_kwargs: Optional[Dict[str, Any]] = None,
            verbose: int = 0,
            seed: Optional[int] = None,
            device: Union[th.device, str] = "auto",
            _init_setup_model: bool = True,
    ):

        super(RecurrentSAC, self).__init__(
            policy,
            env,
            LSTMPolicy,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            create_eval_env=create_eval_env,
            seed=seed,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            use_sde_at_warmup=use_sde_at_warmup,
            optimize_memory_usage=optimize_memory_usage,
            supported_action_spaces=(gym.spaces.Box),
        )

        self.target_entropy = target_entropy
        self.log_ent_coef = None  # type: Optional[th.Tensor]
        # Entropy coefficient / Entropy temperature
        # Inverse of the reward scale
        self.ent_coef = ent_coef
        self.target_update_interval = target_update_interval
        self.ent_coef_optimizer = None

        self.normalize_advantage = normalize_advantage
        self._last_lstm_states = None

        if _init_setup_model:
            self._setup_model()

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

    def _create_aliases(self) -> None:
        self.actor = self.policy.actor
        self.critic = self.policy.critic
        self.critic_target = self.policy.critic_target

    def collect_rollouts(
        self,
        data_in,
        reward_list,
        reward_avg_list,
        step_list,
        cost_list,
        env: VecEnv,
        callback: BaseCallback,
        reward_type: str,
        mode: str,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
        cau: str = "cau",
        sem: str = "cls",
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
        assert isinstance(
            rollout_buffer, (RecurrentRolloutBuffer, RecurrentDictRolloutBuffer)
        ), f"{rollout_buffer} doesn't support recurrent policy"

        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        episode_rewards, total_timesteps = [], []
        episode_reward = 0.
        global spec, stl_buf, signal_buf

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        lstm_states = deepcopy(self._last_lstm_states)

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                episode_starts = th.tensor(self._last_episode_starts, dtype=th.float32, device=self.device)
                actions, values, log_probs, lstm_states = self.policy.forward(obs_tensor, lstm_states, episode_starts)

            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions
            # Clip the actions to avoid out of bound error
            if isinstance(self.action_space, spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)
            name = self.env.envs[0].unwrapped.spec.id
            cost = 0

            if name == "CartPole-v1" and reward_type == "STL":
                # print(env.envs[0].observation_space)
                # print(new_obs)
                # obs = new_obs[0][-4:]
                obs = new_obs[0]
                loc = abs(obs[0])
                speed = abs(obs[1])
                angle = abs(obs[2])
                velocity = abs(obs[3])
                # print(obs[0])

                u = clipped_actions
                if callback.ctr == 0:
                    if sem == "cau":
                        signal_buf = 'loc,speed,angle,velocity'
                        stl_buf = 'alw_[0,1500](speed[t]<0.2 and angle[t]<0.02 and loc[t]<0.5)'
                    # try:
                    #     spec.parse()
                    #     spec.pastify()
                    # except rtamt.RTAMTException as err:
                    #     print('RTAMT Exception: {}'.format(err))
                    #     sys.exit()
                # if sem == "agm":
                #     rob = spec.update(callback.ctr, [('loc', loc), ('speed', speed), ('angle', angle), ('velocity', velocity)])
                if sem == "cau":
                    data = np.array([callback.ctr, loc, speed, angle, velocity])
                    # print(data)
                    data_in.append(data)
                    if cau == "cau":
                        robs = stl_causation_opt.mainFunction(signal_buf, stl_buf, np.array(data_in), np.array([0]))
                    elif cau == "cau_app":
                        robs = stl_causation_app.mainFunction(signal_buf, stl_buf, np.array(data_in), np.array([0]))
                    elif cau == "rob":
                        robs = stl_robustness.mainFunction(signal_buf, stl_buf, np.array(data_in), np.array([0]))
                    elif cau == "lse":
                        robs = stl_robustness_lse.mainFunction(signal_buf, stl_buf, np.array(data_in), np.array([0]))
                    elif cau == "sss":
                        robs = stl_robustness_sss.mainFunction(signal_buf, stl_buf, np.array(data_in), np.array([0]))


                    if len(data_in)>=6:
                        rob = robs[int(len(data_in)) - 1]
                    else:
                        rob = 0.2
                    # k = len(data_in)
                    # rob = sum((j > 0) for j in robs[0:k]) / data_in.maxlen
                    # if len(data_in)>=3:
                    #     rob = robs[int(len(data_in)) - 1] - robs[int(len(data_in)) - 2]
                    # elif len(data_in) >= 2:
                    #     rob = robs[int(len(data_in)) - 1]
                    # else:
                    #     rob = 0
                # else:
                #     rob = spec.update(callback.ctr, [('loc', loc), ('speed', speed), ('angle', angle), ('velocity', velocity)])
                rewards = np.array([rob])

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if not callback.on_step():
                return False

            reward_list[-1] += rewards[0]
            step_list[-1] += 1
            cost_list[-1] += cost

            self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done_ in enumerate(dones):
                if (
                    done_
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_lstm_state = (
                            lstm_states.vf[0][:, idx : idx + 1, :].contiguous(),
                            lstm_states.vf[1][:, idx : idx + 1, :].contiguous(),
                        )
                        # terminal_lstm_state = None
                        episode_starts = th.tensor([False], dtype=th.float32, device=self.device)
                        terminal_value = self.policy.predict_values(terminal_obs, terminal_lstm_state, episode_starts)[0]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(
                self._last_obs,
                actions,
                rewards,
                self._last_episode_starts,
                values,
                log_probs,
                lstm_states=self._last_lstm_states,
            )

            self._last_obs = new_obs
            self._last_episode_starts = dones
            self._last_lstm_states = lstm_states

            if dones[0]:
                reward_list.append(0)
                step_list.append(0)
                cost_list.append(0)
                if self._episode_num > 10:
                    avg_reward = np.mean(reward_list[-10:])
                else:
                    avg_reward = None
                reward_avg_list.append(avg_reward)
                print(reward_list[self._episode_num], callback.ctr)
                self._episode_num += 1
                callback.ctr = 0
                data_in.clear()

        with th.no_grad():
            # Compute value for the last timestep
            episode_starts = th.tensor(dones, dtype=th.float32, device=self.device)
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device), lstm_states.vf, episode_starts)

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.on_rollout_end()

        return True

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizers learning rate
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            # We need to sample because `log_std` may have changed between two gradient steps
            if self.use_sde:
                self.actor.reset_noise()

            # Action by the current actor for the sampled state
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None:
                # Important: detach the variable from the graph
                # so we don't change it with other losses
                # see https://github.com/rail-berkeley/softlearning/issues/60
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())

            # Optimize entropy coefficient, also called
            # entropy temperature or alpha in the paper
            if ent_coef_loss is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with th.no_grad():
                # Select action according to policy
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                # Compute the next Q values: min over all critics targets
                next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                # add entropy term
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                # td error + entropy term
                target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Get current Q-values estimates for each critic network
            # using action from the replay buffer
            current_q_values = self.critic(replay_data.observations, replay_data.actions)

            # Compute critic loss
            critic_loss = 0.5 * sum([F.mse_loss(current_q, target_q_values) for current_q in current_q_values])
            critic_losses.append(critic_loss.item())

            # Optimize the critic
            self.critic.optimizer.zero_grad()
            # print(critic_loss.grad)
            critic_loss.backward()

            '''
            ######
            #from torch.autograd import gradcheck
            #test = gradcheck(critic_loss1, current_q_values, eps=1e-6, atol=1e-4)
            #print(test)
            ######

            #print(self.critic.optimizer.param_groups[0]['params'][0].grad)
            #print(len(self.critic.optimizer.param_groups))
            lp=len(self.critic.optimizer.param_groups[0]['params'])
            #print(len(self.critic.optimizer.param_groups[0]['params'][0].grad))
            mean_grad_1=th.mean(th.abs(self.critic.optimizer.param_groups[0]['params'][0].grad))
            min_grad_1=th.min(th.abs(self.critic.optimizer.param_groups[0]['params'][0].grad))
            min_grad_all=[]
            for i in range(lp):
                min_grad_all.append(th.min(th.abs(self.critic.optimizer.param_groups[0]['params'][i].grad)))
            min_grad=np.min(min_grad_all)
            mean_grad_all=[]
            for i in range(lp):
                mean_grad_all.append(th.min(th.abs(self.critic.optimizer.param_groups[0]['params'][i].grad)))
            mean_grad=np.mean(mean_grad_all)


            grad_all=[]
            for i in range(lp):
                grad_all.append(th.all(~th.isnan(self.critic.optimizer.param_groups[0]['params'][i].grad)))
            mylog=[min_grad_1.numpy(),mean_grad_1.numpy(),min_grad,mean_grad]
            mylog=[grad_all]
            #print(torch.all(~torch.isnan(h)))
            with open("grad.csv", 'a') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow(mylog)

            #print(self.critic.optimizer.param_groups[0]['params'])
            #print("critic data[0] ",replay_data.observations[0])
            #print(replay_data.observations.grad)
            #exit()
            '''

            self.critic.optimizer.step()

            # Compute actor loss
            # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
            # Mean over all critic networks
            q_values_pi = th.cat(self.critic.forward(replay_data.observations, actions_pi), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
            actor_losses.append(actor_loss.item())

            # Optimize the actor
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))

    def learn(
        self: SelfRecurrentPPO,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        eval_env: Optional[GymEnv] = None,
        eval_freq: int = -1,
        n_eval_episodes: int = 5,
        tb_log_name: str = "RecurrentPPO",
        eval_log_path: Optional[str] = None,
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
        reward_type: str = "nominal",
        mode: str = "train",
        sem: str = "cls",
        cau: str = "cau",
        run: str = "test",
    ) -> SelfRecurrentPPO:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            eval_env=eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            tb_log_name=tb_log_name,
            eval_log_path=eval_log_path,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
            reward_type=reward_type,
            mode=mode,
            sem=sem,
            cau=cau,
            run=run,
        )

    def learn(
            self,
            total_timesteps: int,
            callback: MaybeCallback = None,
            log_interval: int = 4,
            eval_env: Optional[GymEnv] = None,
            eval_freq: int = -1,
            n_eval_episodes: int = 5,
            tb_log_name: str = "SAC",
            eval_log_path: Optional[str] = None,
            reset_num_timesteps: bool = True,
            reward_type: str = "nominal",
            sem: str = "cls",
            cau: str = "cau",
            run: str = "test",
            data_len: int = 1,
    ) -> OffPolicyAlgorithm:

        return super(SAC, self).learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            eval_env=eval_env,
            eval_freq=eval_freq,
            n_eval_episodes=n_eval_episodes,
            tb_log_name=tb_log_name,
            eval_log_path=eval_log_path,
            reset_num_timesteps=reset_num_timesteps,
            reward_type=reward_type,
            sem=sem,
            cau=cau,
            run=run,
            data_len=data_len,
        )

    def _excluded_save_params(self) -> List[str]:
        return super(SAC, self)._excluded_save_params() + ["actor", "critic", "critic_target"]

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "actor.optimizer", "critic.optimizer"]
        if self.ent_coef_optimizer is not None:
            saved_pytorch_variables = ["log_ent_coef"]
            state_dicts.append("ent_coef_optimizer")
        else:
            saved_pytorch_variables = ["ent_coef_tensor"]
        return state_dicts, saved_pytorch_variables
