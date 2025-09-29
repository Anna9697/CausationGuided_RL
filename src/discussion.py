import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os


env_list=['/ppo_cartpole_tensorboard', '/ppo_pointgoal1_tensorboard', '/sac_hopper_tensorboard', '/sac_walker_tensorboard']
cau_list=['./log_stl_cau_app', './log_stl_sss', './log_stl_lse', './log_stl_rob']
# logdir="./log_6_cau_app/ppo_pointgoal1_tensorboard/PPO_10/"
for env in env_list:
    cau_result = []
    for cau in cau_list:
        temp=[]
        for i in range(10):
            if env == '/ppo_cartpole_tensorboard' or env == '/ppo_pointgoal1_tensorboard':
                model = '/PPO_'
            else:
                model = '/SAC_'
            logdir = cau + env + model + str(i+1) + '/'
            event_files = [f for f in os.listdir(logdir) if f.startswith("events.out")]
            event_path = os.path.join(logdir, event_files[0])

            ea = EventAccumulator(event_path)
            ea.Reload()

            # 获取 wall_time 时间戳（以秒为单位）
            scalars = ea.Scalars("rollout/ep_len_mean")

            start_time = scalars[0].wall_time
            end_time = scalars[-1].wall_time

            # scalars_step = ea.Scalars("timesteps")
            steps = [s.step for s in scalars]
            final_step = max(steps)
            time_perstep = (end_time - start_time)/final_step
            temp.append(time_perstep)
        mean = np.mean(temp)
        std = np.std(temp)
        cau_result.append([mean, std])
    print(env+':', cau_result)

# env_list=['/ppo_cartpole_tensorboard', '/sac_hop_tensorboard']
env_log = '/sac_hopper_tensorboard'
tau_list=['./log_stl_cau_app', './log_tau_21', './log_tau_26', './log_tau_31', './log_tau_36']
tau_result = []
for tau in tau_list:
    temp=[]
    for i in range(10):
        if env_log == '/ppo_cartpole_tensorboard':
            model = '/PPO_'
        else:
            model = '/SAC_'
        logdir = tau + env_log + model + str(i+1) + '/'
        event_files = [f for f in os.listdir(logdir) if f.startswith("events.out")]
        event_path = os.path.join(logdir, event_files[0])

        ea = EventAccumulator(event_path)
        ea.Reload()

        # 获取 wall_time 时间戳（以秒为单位）
        scalars = ea.Scalars("rollout/ep_len_mean")

        start_time = scalars[0].wall_time
        end_time = scalars[-1].wall_time

        # scalars_step = ea.Scalars("timesteps")
        steps = [s.step for s in scalars]
        final_step = max(steps)
        time_perstep = (end_time - start_time)/final_step
        temp.append(time_perstep)
    mean = np.mean(temp)
    std = np.std(temp)
    tau_result.append([mean, std])
print('tau_'+env_log+':', tau_result)

env_log = '/ppo_cartpole_tensorboard'
tau_list=['./log_stl_cau_app', './log_tau_16', './log_tau_21', './log_tau_26', './log_tau_31']
tau_result = []
for tau in tau_list:
    temp=[]
    for i in range(10):
        if env_log == '/ppo_cartpole_tensorboard':
            model = '/PPO_'
        else:
            model = '/SAC_'
        logdir = tau + env_log + model + str(i+1) + '/'
        event_files = [f for f in os.listdir(logdir) if f.startswith("events.out")]
        event_path = os.path.join(logdir, event_files[0])

        ea = EventAccumulator(event_path)
        ea.Reload()

        # 获取 wall_time 时间戳（以秒为单位）
        scalars = ea.Scalars("rollout/ep_len_mean")

        start_time = scalars[0].wall_time
        end_time = scalars[-1].wall_time

        # scalars_step = ea.Scalars("timesteps")
        steps = [s.step for s in scalars]
        final_step = max(steps)
        time_perstep = (end_time - start_time)/final_step
        temp.append(time_perstep)
    mean = np.mean(temp)
    std = np.std(temp)
    tau_result.append([mean, std])
print('tau_'+env_log+':', tau_result)