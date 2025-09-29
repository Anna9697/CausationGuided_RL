# Load the trained agent
# some_file.py
import sys
import numpy as np
import pickle
import warnings
import argparse
warnings.filterwarnings("ignore")
import safety_gym
from safety_gym.envs.engine import Engine
from gym.envs.registration import register
from env.GymSeq import GymSeq
import highway_env
# insert at 1, 0 is the script path (or '' in REPL)
#sys.path.insert(1, '/home/nikhil/RESEARCH/RL/stable-baselines3/')
import stl_robustness
import stl_causation_app
import stl_causation_opt


import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np
import psutil, os, time
import pandas as pd
import pickle

import gym
from stable_baselines3 import SAC
from stable_baselines3 import PPO
from stable_baselines3 import DDPG
from stable_baselines3.common.env_util import make_vec_env 
from stable_baselines3.common.vec_env import DummyVecEnv
from scipy.spatial.transform import Rotation as R
#modelid=4

from highway_env.vehicle.controller import ControlledVehicle


def register_env(env_id, sem, tau):
    open('gvars.py', 'w').close()
    file1 = open('gvars.py', 'w')
    s = "rob_type=1"
    file1.write(s)
    file1.close()
    if env_id == "Hopper-v3":
        env_pre = gym.make('Hopper-v3')
        if sem == 'cau' or sem == 'cau_app':
            if tau == 0:
                env = GymSeq(env_pre, n_steps=16)
            else:
                env = GymSeq(env_pre, n_steps=tau)
        else:
            env = GymSeq(env_pre, n_steps=1)
    elif env_id == "Walker2d-v3":
        # env = gym.make('Walker2d-v3')
        env_pre = gym.make('Walker2d-v3')
        if sem == 'cau' or sem == 'cau_app':
            env = GymSeq(env_pre, n_steps=16)
        else:
            env = GymSeq(env_pre, n_steps=1)
    elif env_id == "CartPole-v1":
        env_pre = gym.make('CartPole-v1')
        if sem == 'cau' or sem == 'cau_app':
            if tau == 0:
                env = GymSeq(env_pre, n_steps=11)
            else:
                env = GymSeq(env_pre, n_steps=tau)
        else:
            env = GymSeq(env_pre, n_steps=1)
    elif env_id == "PointGoal1-v0":
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
        env_pre = gym.make('PointGoal1-v0')
        if sem == 'cau' or sem == 'cau_app':
            env = GymSeq(env_pre, n_steps=41)
        else:
            env = GymSeq(env_pre, n_steps=1)
        # env = gym.make('PointGoal1-v0')
    return env

def compute_cost(env, env_id, filename):
    #evaluating w.r.t. classical semantics
    #if modelid=="HC":
    #    print(modelid)
    #    exit()
    open('gvars.py', 'w').close()
    file1 = open('gvars.py', 'w')
    s="rob_type=1"
    file1.write(s)
    file1.close()

    tsc = np.array([]) #total state cost
    tcc = np.array([])  # total control cost
    tsp = np.array([])  # total control cost
    tsteps=np.array([])
    tdist=np.array([])
    tdr=np.array([])
    tseeds=100
    avg_min_rob=[]
    avg_mean_rob=[]
    cost = 0
    SAT = 0
    FULL_SAT = 0
    signal_buf=''
    stl_buf=''
    sat_buf=''
    total_step=0
    for sed in range(0,tseeds):
        env.seed(sed)
        sc = np.array([])
        steps = 0
        sp = np.array([])
        cc = np.array([])
        if env_id == "PointGoal1-v0" or env_id=="CartPole-v1":
        # if modelid=="Safexp-PointGoal1-v0" or modelid=="PointGoal1-v0" or modelid=="PointButton1-v0" or modelid=="ZigZag-v0":
            model = PPO.load("result/"+filename)
        else:
            model = SAC.load("result/"+filename)

        nsteps = 1000

        #############

        # Enjoy trained agent
        obs = env.reset()
        # print(sorted(env.obs_space_dict.keys()))
        # print(obs)
        # print("obs_space_dict", env.obs_space_dict)
        min_rob=1000
        mean_rob=[]
        dr=0
        rob=0
        data_in=[]
        # cost = 0
        pass_d1 = False
        pass_d2 = False
        pass_d3 = False
        pass_d4 = False

        x=0

        for i in range(nsteps):
            #action, _states = model.predict(obs, deterministic=True)
            action, _states = model.predict(obs)
            # print(obs)
            # print(i)
            obs, rewards, dones, info = env.step(action)
            total_step += 1
            # env.render()

            # if i % 100 == 0:
            #     tag = f"After {100} steps"
            #     process = psutil.Process(os.getpid())
            #     mem_mb = process.memory_info().rss / 1024 / 1024
            #     print(f"[{tag}] Memory usage: {mem_mb:.2f} MB")

            if env_id=="Hopper-v3":
                obs_new = obs[-11:]
                x = info['x_position']
                #if x>50:
                #    break
                sc=np.append(sc,50-x)
                p = info['x_position']
                v = info['x_velocity']
                z , a = obs_new[0:2]

                if i==0:
                    signal_buf = 'p,v,z,a'
                    stl_buf = 'alw_[0,1000](ev_[0,15](v[t]>0.5) and (z[t]>0.7) and (a[t]<1))'
                    sat_buf = 'alw_[0,1000]((z[t]>0.7) and (a[t]<1))'
                data = np.array([i, p, v, z, abs(a)])
                data_in.append(data)
                min_rob=min(min_rob,rob)
                mean_rob.append(rob)
                if z <= 0.7 or abs(a) >= 1.0:
                    cost += 1
                else:
                    if dones and i < 999:
                        print(i)
                        cost += 1

            elif env_id=="Walker2d-v3":
                x = info['x_position']
                #if x>50:
                #    break
                # print(obs)
                obs_new = obs[-17:]
                sc=np.append(sc,50-x)
                px = info['x_position']
                vx = info['x_velocity']
                z , a = obs_new[0:2]
                # print(abs(z-0.6), abs(a))

                if i==0:
                    signal_buf = 'px,vx,z,a'
                    stl_buf = 'alw_[0,1500](ev_[0,15](vx[t]>0.5) and (z[t]<0.6) and (a[t]<1))'
                    # sat_buf = 'alw_[0,1500]((z[t]<1.4) and (a[t]<1))'
                    sat_buf = 'alw_[0,1500]((z[t]<0.6) and (a[t]<1))'
                # print(px,vx,z,a)
                data = np.array([i, px, vx, abs(z - 1.4), abs(a)])
                data_in.append(data)
                # robs = stl_causation_app.mainFunction(signal_buf, stl_buf, np.array(data_in), np.array([0]))

                min_rob=min(min_rob,rob)
                mean_rob.append(rob)
                # if i == nsteps - 1 or dones:
                #     robs = stl_causation_opt.mainFunction(signal_buf, stl_buf, np.array(data_in), np.array([0]))
                # # if rob<0:
                if abs(z - 1.4) >= 0.6 or abs(a) >= 1.0:
                    cost += 1
                else:
                    if dones and i < 999:
                        print(i)
                        cost += 1
                #     cost = sum((j <= 0) for j in robs[0:(len(data_in)-1)])

            elif env_id=="CartPole-v1":
                obs_new = obs[-4:]
                loc = abs(obs_new[0])
                speed = abs(obs_new[1])
                angle = abs(obs_new[2])
                velocity = abs(obs_new[3])
                x = loc
                # print(loc,speed, angle)

                if i==0:
                    signal_buf = 'loc,speed,angle,velocity'
                    # stl_buf = 'alw_[0,500](alw_[0,5](loc[t]<0.5 and speed[t]<0.1 and angle[t]<0.02))'
                    stl_buf = 'alw_[0,500](ev_[0,10](speed[t]<0.1) and loc[t]<0.5 and angle[t]<0.1)'
                    sat_buf = 'alw_[0,500](loc[t]<0.5 and angle[t]<0.1)'
                data = np.array([i, loc, speed, angle, velocity])
                data_in.append(data)
                min_rob=min(min_rob,rob)
                mean_rob.append(rob)
                if loc >= 0.5 or angle >= 0.1:
                    # print([loc, angle])
                    cost += 1
                else:
                    if dones and i < 499:
                        print(i)
                        cost += 1

            elif env_id=="PointGoal1-v0":
                # dg = 0.
                # dc = 0.
                obs_new = obs[-44:]
                dg = max(obs_new[3:19])
                dc = max(obs_new[22:38])
                # for d1 in obs_new[3:19]:
                #     if d1>=0. and d1<=1. and d1 > dg:
                #         dg = d1
                # for d2 in obs_new[22:38]:
                #     if d2>=0. and d2<=1. and d2 > dc:
                #         dc = d2
                Dg = (1 - dg) * env.lidar_max_dist
                Dc = (1 - dc) * env.lidar_max_dist
                x=Dg
                if i==0:
                    signal_buf = 'dg,dc'
                    # stl_buf = 'alw_[0,1500](ev_[0,40](dg[t]<=0.3) and dc[t]>0.2)'
                    stl_buf = 'alw_[0,1500](dc[t]<0.96667 and ev_[0,40](dg[t]>=0.95))'
                    # sat_buf = 'alw_[0,1500](dc[t]>0.2)'
                    sat_buf = 'alw_[0,1500](dc[t]<0.96667)'
                data = np.array([i, dg, dc])
                # data = np.array([i, Dg, Dc])
                data_in.append(data)
                min_rob=min(min_rob,rob)
                mean_rob.append(rob)
                if dc>=0.96667:
                    cost += 1
                else:
                    if dones and i < 999:
                        print(i)
                        cost += 1

            #sp=np.append(sp,x)
            #cc=np.append(cc,action)
            cc=np.append(cc,np.square(action).sum())
            dr+=rewards
            if dones == True:
                break
            #print(str(sed)+"   "+str(x)+" "+str(rewards))
            #if dones==True:
        tsc=np.append(tsc,np.sum(np.square(sc)))
        tcc=np.append(tcc,np.sum(np.square(cc)))  
        tdr=np.append(tdr,dr)
        tsteps=np.append(tsteps,steps)  
        tdist=np.append(tdist,x)  
        tmp=np.mean(mean_rob)
        avg_min_rob.append(min_rob)
        avg_mean_rob.append(tmp)
        sat_robs = stl_robustness.mainFunction(signal_buf, sat_buf, np.array(data_in), np.array([0]))
        full_robs = stl_robustness.mainFunction(signal_buf, stl_buf, np.array(data_in), np.array([0]))
        print(sed, len(data_in))

        if env_id == "CartPole-v1":
            # print(sat_robs[len(data_in)-1])
            if len(data_in)>=500 and sat_robs[len(data_in)-1]>0:
                SAT = SAT + 1.
            if len(data_in) >= 500 and full_robs[len(data_in) - 1] > 0:
                FULL_SAT = FULL_SAT + 1.
        else:
            if len(data_in)>=1000 and sat_robs[len(data_in)-1]>0:
                SAT = SAT + 1.

            if len(data_in)>=1000 and full_robs[len(data_in)-1]>0:
                FULL_SAT = FULL_SAT + 1.


    data=np.array(tcc)
    mu=np.mean(data)
    std=np.std(data)
    
    print("-----------------------------------------------")
    print("#### SUMMMARY : CONTROLLER EVALUATION #########")
    print("-----------------------------------------------\n")
    # print("Control Cost (CC) : ",mu,u"\u00B1",std)
    print("Cost Return:", '%.6f'% (cost / total_step))
    print("SAT:", '%.6f'% (SAT / tseeds))
    print("FULL_SAT:", '%.6f' % (FULL_SAT / tseeds))

    data=np.array(tdist)
    std=np.std(data)
    return cost/total_step, SAT/tseeds, FULL_SAT / tseeds



if __name__ == "__main__":  # noqa: C901
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", help="environment ID", type=str, default="Hopper-v3")
    parser.add_argument("--stl", help="reward type", type=str, default="stl")
    parser.add_argument("--sem", help="STL semantics", type=str, default="cau_app")
    parser.add_argument("--seed", help="begin_seed", type=int, default=0)
    parser.add_argument("--model", help="nn_model", type=str, default="ppo")
    parser.add_argument("--tau", help="k", type=int, default=0)

    args = parser.parse_args()
    env_id = args.env
    # file_name =args.file
    stl = args.stl
    sem = args.sem
    seed = args.seed
    model_name = args.model
    tau = args.tau

    env = register_env(env_id, sem, tau)
    cost_list = []
    SAT_list = []
    FSAT_list = []
    for i in range(10):
        file_name = model_name + "_" + env_id + "_" + stl + "_" + sem + "_" + str(seed+i) + ".zip"
        cost, SAT, FULL_SAT = compute_cost(env, env_id, file_name)
        cost_list.append(cost)
        SAT_list.append(SAT)
        FSAT_list.append(FULL_SAT)
    mean_cost = np.mean(cost_list)
    mean_SAT = np.mean(SAT_list)
    mean_FSAT = np.mean(FSAT_list)
    std_cost = np.std(cost_list)
    std_SAT = np.std(SAT_list)
    std_FSAT = np.std(FSAT_list)


    print("#### Finally : CONTROLLER EVALUATION #########")
    print("-----------------------------------------------\n")
    # print("Control Cost (CC) : ",mu,u"\u00B1",std)
    print("Full-SAT List:", FSAT_list)
    print("SAT List:", SAT_list)
    print("CR List:", cost_list)

    print("FULL_SAT:", '%.6f' % (mean_FSAT), "+-", ' %.6f' % (std_FSAT))
    print("SAT:", '%.6f' % (mean_SAT), "+-", ' %.6f' % (std_SAT))
    print("Cost Return:", '%.6f' % (mean_cost), "+-", ' %.6f' % (std_cost))


