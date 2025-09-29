import gym
import torch
from env.GymSeq import GymSeq
from gym.envs.registration import register
import argparse
import warnings
warnings.filterwarnings("ignore")
from stable_baselines3 import SAC, PPO
import timeit

def run_exp(env_id, reward_id, sem, run):
    print(int(run))


    open('gvars.py', 'w').close()
    file1 = open('gvars.py', 'w')
    name="stl"
    if reward_id == 1:
        name="stl"
        s="rob_type=1"
    elif reward_id == 0:
        name="normal"
        s="rob_type=0"

    file1.write(s)
    file1.close()

    reward_type = 'nominal'
    if reward_id == 1:
        reward_type = 'STL'

    if env_id=="Hopper-v3":
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        start = timeit.default_timer()
        env_pre = DummyVecEnv([lambda: gym.make('Hopper-v3')])
        print(env_pre.envs[0])
        if sem == 'cau' or sem == 'cau_app':
            env = GymSeq(env_pre.envs[0], n_steps=16)
        else:
            env = GymSeq(env_pre.envs[0], n_steps=1)

        log_path = "./log_" + name + "_" + str(sem) + "/sac_hopper_tensorboard"
        model = SAC('MlpPolicy', env, learning_starts=3000, use_sde=False,  tensorboard_log=log_path, verbose=1, seed=int(run)%10)
        model.learn(total_timesteps=int(2e6), max_episodes=1600, reward_type=reward_type, stl=name, sem=sem, run=run, data_len=16)
        model.save("./result/sac_Hopper-v3_"+ name + "_" + sem + "_" + str(run))
        end = timeit.default_timer()
        print("training_time is"+str(end-start))
        text_file = open("hopper.txt", "a")
        text_file.write('Time for '+str(run)+' is : '+str(end-start)+'\n')
        text_file.close()


    elif env_id=="Walker2d-v3":
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        start = timeit.default_timer()
        # env = DummyVecEnv([lambda: gym.make('Walker2d-v3')])
        env_pre = DummyVecEnv([lambda: gym.make('Walker2d-v3')])
        print(env_pre.envs[0])
        if sem == 'cau' or sem == 'cau_app':
            env = GymSeq(env_pre.envs[0], n_steps=16)
        else:
            env = GymSeq(env_pre.envs[0], n_steps=1)

        log_path = "./log_" + name + "_" + str(sem) + "/sac_walker_tensorboard"
        model = SAC('MlpPolicy',env,learning_starts=3000, use_sde=False, tensorboard_log=log_path, verbose=1, seed=int(run)%10)
        model.learn(total_timesteps=int(2e6),max_episodes=1600,reward_type=reward_type, stl=name, sem=sem, run=run, data_len=16)

        model.save("./result/sac_Walker2d-v3_" + name + "_" + sem + "_" + str(run))
        end = timeit.default_timer()
        print("training_time is"+str(end-start))
        text_file = open("walker.txt", "a")
        text_file.write('Time for '+str(run)+' is : '+str(end-start)+'\n')
        text_file.close()

    elif env_id=="CartPole-v1":
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        start = timeit.default_timer()
        # env = DummyVecEnv([lambda: gym.make('CartPole-v1')])
        env_pre = DummyVecEnv([lambda: gym.make('CartPole-v1')])
        # print(env_pre.envs[0])
        if sem == 'cau' or sem == 'cau_app':
            env = GymSeq(env_pre.envs[0], n_steps=11)
        else:
            env = GymSeq(env_pre.envs[0], n_steps=1)

        log_path = "./log_" + name + "_" + str(sem) + "/ppo_cartpole_tensorboard"
        model = PPO('MlpPolicy', env, tensorboard_log=log_path, learning_rate=4e-4, verbose=1, seed=int(run)%10+30)
        model.learn(total_timesteps=int(1e6), max_episodes=500, reward_type=reward_type, stl=name, sem=sem, run=run, data_len=11)

        model.save("./result/ppo_CartPole-v1_"+name+"_"+sem+"_"+str(run))
        end = timeit.default_timer()
        print("training_time is"+str(end-start))
        text_file = open("cartpole.txt", "a")
        text_file.write('Time for '+str(run)+' is : '+str(end-start)+'\n')
        text_file.close()


    elif env_id=="PointGoal1-v0":
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        start = timeit.default_timer()

        # env = DummyVecEnv([lambda: gym.make('PointGoal1-v0')])
        env_pre = DummyVecEnv([lambda: gym.make('PointGoal1-v0')])
        if sem == 'cau' or sem == 'cau_app':
            env = GymSeq(env_pre.envs[0], n_steps=41)
        else:
            env = GymSeq(env_pre.envs[0], n_steps=1)

        policy_kwargs = dict(net_arch=[dict(pi=[1024, 512, 512], vf=[1024, 512, 512])])
        log_path = "./log_" + name + "_" + str(sem) + "/ppo_pointgoal1_tensorboard"
        model = PPO('MlpPolicy', env, policy_kwargs=policy_kwargs, tensorboard_log=log_path, learning_rate=1e-5, ent_coef=0.0, verbose=1, seed=int(run)%10)
        if torch.cuda.device_count() > 1:
            print(f"Using {torch.cuda.device_count()} GPUs!")
            model.policy = torch.nn.DataParallel(model.policy)
        else:
            print(f"Using 1 GPUs!")
        model.learn(total_timesteps=int(5e5), reward_type=reward_type, stl=name, sem=sem, run=run, data_len=41)
        model.save("./result/ppo_PointGoal1-v0_"+name+"_"+sem+"_"+str(run))
        end = timeit.default_timer()
        print("training_time is"+str(end-start))
        text_file = open("pointgoal1.txt", "a")
        text_file.write('Time for '+str(run)+' is : '+str(end-start)+'\n')
        text_file.close()

    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", help="environment ID", type=str, default="Hopper-v3")
    parser.add_argument("--reward", help="reward type", default=1, type=int, required=False)
    parser.add_argument("--sem", help="STL semantics", default="cau", type=str, required=False)
    parser.add_argument("--run", help="run id", default="00", type=str, required=False)
  
    args = parser.parse_args()

    env_id = args.env
    reward = args.reward
    run = args.run
    sem = args.sem

    config = {
        'robot_base': 'xmls/point.xml',
        'task': 'goal',
        'observe_goal_lidar': True,
        'observe_hazards': True,
        'observe_vases': False,
        'constrain_hazards': True,
        'lidar_max_dist': 6,
        'lidar_num_bins': 16,
        'hazards_num': 9,
        'vases_num': 0,
        'placements_extents': [-1.5, -1.5, 1.5, 1.5],
        'hazards_locations': [[1.34, 1.2], [1.2, -1.0], [0.45, 0.67], [1.26, 0.33], [-0.23, -0.4], [0.5, -1.27],
                              [1.7, -0.14], [-0.77, -1.15], [-0.52, 0.43]],
        'robot_placements': [[-2.5, -2.5, 2.5, 2.5]],
        'goal_locations': [[0.6, 0]],
        'goal_size': 0.3,
        'goal_keepout': 0.305,
        'hazards_size': 0.2,
        'hazards_keepout': 0.2,
        'continue_goal': False
    }
    register(id='PointGoal1-v0',
             entry_point='safety_gym.envs.mujoco:Engine',
             kwargs={'config': config})


    for i in range(10):
        run_exp(env_id, reward, sem, str(int(run)+i))
        print("--------------finish for 1 episode---------------")
