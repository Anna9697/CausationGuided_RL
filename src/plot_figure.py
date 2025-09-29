import gym
# import env.zigzag as zigzag

# from safety_gym.envs.engine import Engine
import sys
# sys.path.insert(1, '/home/nikhil/RESEARCH/RL/SSFC/')
# from pathlib import Path
# sys.path.append(r"/home/tang/safety-gym/safety_gym/envs")
import safety_gym
from safety_gym.envs.engine import Engine
from gym.envs.registration import register
import argparse
import warnings
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Rectangle
from matplotlib import rcParams


# from proplot import rc
warnings.filterwarnings("ignore")


import numpy as np
import pandas as pd
import seaborn as sns
import math
from stable_baselines3 import SAC
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import timeit
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
import pickle

def plot_seaborn(env_name, df, fign, save_fig=False):
    from matplotlib.pyplot import MultipleLocator
    font_legend = {'family': 'serif',
             'weight': 'normal',
             'size': 36,
    }
    font_legend1 = {'family': 'serif',
                   'weight': 'normal',
                   'size': 26,
                   }

    # else:
    plt.figure(figsize=(10, 6))
    colors = ['#be0e23', '#5f89b1', '#d89c7c', '#81b095']
    ax = sns.lineplot(x="Episode", y="Reward", hue="cau", data=df, ci="sd", legend="full", palette=colors, style="cau")

    h,_ = ax.get_legend_handles_labels()
    # 添加标题和坐标轴标签
    if env_name == 'PointGoal1-v0' or env_name == 'highway-v0':
        plt.legend(h,["CAU", "SSS", "LSE", "CLS"], ncol=1, prop=font_legend1, loc='lower right', labelspacing=0.3, handletextpad=0.1)
    else:
        plt.legend(h, ["CAU", "SSS", "LSE", "CLS"], ncol=1, prop=font_legend1, loc='upper left', labelspacing=0.3, handletextpad=0.1)

    if env_name == 'Hopper-v3' or env_name=='Walker2d-v3':
        x_major_locator = MultipleLocator(400)
        ax.xaxis.set_major_locator(x_major_locator)
    if env_name == 'PointGoal1-v0':
        plt.ylim(-1.0, 0.18)
    if env_name == 'Hopper-v3':
        plt.ylim(-0.2, 1.15)
    if env_name == 'Walker2d-v3':
        y_major_locator = MultipleLocator(0.2)
        ax.yaxis.set_major_locator(y_major_locator)
        plt.ylim(-0.9, 1.1)

    plt.xticks(fontproperties='Times New Roman', size=28)
    plt.yticks(fontproperties='Times New Roman', size=28)
    plt.ylabel("Average Reward", font_legend)
    plt.xlabel("Episode", font_legend)

    # if env_name == "PointGoal1-v0":
    #     ax_inset = inset_axes(ax, width="50%", height="50%", loc="lower right")
    #     df_tail = df[df["Episode"] > 400]
    #     sns.lineplot(x="Episode", y="Reward", hue="cau", data=df_tail, ci="sd", palette=colors, ax=ax_inset,
    #                  legend=False, style="cau")
    #     ax_inset.set_xlim(400, 500)  # 放大范围
    #     ax_inset.set_ylim(-0.1, 0.16)  # 根据你的需求调整
    #     ax_inset.xaxis.set_label_position('top')
    #     ax_inset.xaxis.tick_top()
    #     ax_inset.set_xlabel("")
    #     ax_inset.set_ylabel("")
    #     mark_inset(ax, ax_inset, loc1=2, loc2=1, fc="none", ec="0.3", linestyle="--")


    if save_fig:
        plt.savefig(fign, bbox_inches='tight', pad_inches=0.01)
    plt.show()

def plot_compare(env_name, df, fign, save_fig=False):
    from matplotlib.pyplot import MultipleLocator
    font_legend = {'family': 'serif',
             'weight': 'normal',
             'size': 36,
    }
    font_legend1 = {'family': 'serif',
                   'weight': 'normal',
                   'size': 26,
                   }

    # else:
    plt.figure(figsize=(10, 6))
    # colors = ['#be0e23', '#b092b6']
    colors = ['#be0e23', '#7c7cba']
    linestyles = {'cau_app': (), 'none': (5, 2, 5, 2, 1, 2)}
    ax = sns.lineplot(x="Episode", y="Reward", hue="cau", data=df, ci="sd", legend="full", palette=colors, style="cau", dashes=linestyles)

    h,_ = ax.get_legend_handles_labels()
    # 添加标题和坐标轴标签
    if env_name == 'PointGoal1-v0':
        plt.legend(h, ["CAU", "BAS"], ncol=1, prop=font_legend1, loc='lower right', labelspacing=0.3, handletextpad=0.1)
        # plt.legend(h, ["CAU", "Vanilla"], ncol=1, prop=font_legend1, loc='upper left', labelspacing=0.3, handletextpad=0.1)
    else:
        plt.legend(h, ["CAU", "BAS"], ncol=1, prop=font_legend1, loc='upper left', labelspacing=0.3, handletextpad=0.1)

    if env_name == 'Hopper-v3' or env_name=='Walker2d-v3':
        x_major_locator = MultipleLocator(400)
        ax.xaxis.set_major_locator(x_major_locator)
    if env_name == 'PointGoal1-v0':
        plt.ylim(-1.0, 0.18)
    if env_name == 'Hopper-v3':
        plt.ylim(-0.1, 1.15)
    if env_name == 'Walker2d-v3':
        y_major_locator = MultipleLocator(0.2)
        ax.yaxis.set_major_locator(y_major_locator)
        plt.ylim(-0.1, 1.1)

    plt.xticks(fontproperties='Times New Roman', size=28)
    plt.yticks(fontproperties='Times New Roman', size=28)
    plt.ylabel("Average Reward", font_legend)
    plt.xlabel("Episode", font_legend)

    if save_fig:
        plt.savefig(fign, bbox_inches='tight', pad_inches=0.01)
    plt.show()



# def zero_normalize(dataset):
#     reward_min = np.min(dataset)
#     reward_max = np.max(dataset)
#     normal = max(abs(reward_min), abs(reward_max))
#     normalized_data = (dataset) / normal
#     # result = all(value <= 1.0 for value in normalized_data)
#     # print(result)
#     return normalized_data

if __name__ == "__main__":  # noqa: C901
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", help="environment ID", type=str, default="Hopper-v3")
    parser.add_argument("--stl", help="reward type", default="stl", type=str, required=False)
    parser.add_argument("--sem", help="STL semantics", default="cau", type=str, required=False)
    parser.add_argument("--run", help="run id", default="00", type=str, required=False)
    parser.add_argument("--mode", help="nn_model", default="single", type=str, required=False)

    args = parser.parse_args()

    env_id = args.env
    run = args.run
    # stl = args.stl
    mode = args.mode


    if mode == "seaborn":
        fign_seaborn = "./result/fig/" + "seaborn_" + env_id + "_" + run + ".pdf"
        cau_list = [["stl", "cau_app"], ["stl", "sss"], ["stl", "lse"], ["stl", "rob"]]
        data = []

        # run_list = [1,3,5,7,9]

        for j in range(len(cau_list)):
            all_rewards = []
            all_rewards_norm = []
            reward_min = 1000
            reward_max = -1000
            sem = cau_list[j][0]
            cau1 = cau_list[j][1]
            for i in range(10):
                reward_file_name = "./result/list/" + env_id + "_reward_avg_" + sem + "_" + cau1 + "_" + str(int(run)+i) + ".pkl"
                with open(reward_file_name, "rb") as f:
                    rewards = pickle.load(f)
                reward_min = min(reward_min, np.min(rewards))
                reward_max = max(reward_max, np.max(rewards))
                all_rewards.append(rewards)
            # if env_id == "highway-v0":
            #     reward_min = -20.0
            for reward_list in all_rewards:
                if env_id == "CartPole-v1":
                    reward_norm = (reward_list[1:500])/max(abs(reward_min), abs(reward_max))
                else:
                    reward_norm = (reward_list[1:]) / max(abs(reward_min), abs(reward_max))
                all_rewards_norm.append(reward_norm)

            data.append(pd.DataFrame(all_rewards_norm, columns=range(len(reward_norm))).melt(var_name='Episode', value_name='Reward'))
            data[j]['cau'] = cau1
        df = pd.concat(data, ignore_index=True)
        plot_seaborn(env_id, df, fign_seaborn, save_fig=True)
    
    elif mode == "compare":
        fign_seaborn = "./result/fig/" + "compare_" + env_id + "_" + run + ".pdf"
        cau_list = [["stl", "cau_app"], ["normal", "none"]]
        data = []

        for j in range(len(cau_list)):
            all_rewards = []
            all_rewards_norm = []
            reward_min = 1000
            reward_max = -1000
            sem = cau_list[j][0]
            cau1 = cau_list[j][1]
            for i in range(10):
                reward_file_name = "./result/list/" + env_id + "_reward_avg_" + sem + "_" + cau1 + "_" + str(int(run)+i) + ".pkl"
                with open(reward_file_name, "rb") as f:
                    rewards = pickle.load(f)
                reward_min = min(reward_min, np.min(rewards))
                reward_max = max(reward_max, np.max(rewards))
                if sem == "normal":
                    # print(rewards[0])
                    rewards_flat = list(np.array(rewards).flatten())
                    rewards = rewards_flat
                all_rewards.append(rewards)
            for reward_list in all_rewards:
                if env_id == "CartPole-v1":
                    reward_norm = (reward_list[1:500])/max(abs(reward_min), abs(reward_max))
                else:
                    reward_norm = (reward_list[1:]) / max(abs(reward_min), abs(reward_max))
                if env_id == "PointGoal1-v0" and j==1:
                    reward_norm = list(np.array(reward_norm) - 0.88)
                all_rewards_norm.append(reward_norm)

            data.append(pd.DataFrame(all_rewards_norm, columns=range(len(reward_norm))).melt(var_name='Episode', value_name='Reward'))
            data[j]['cau'] = cau1
        df = pd.concat(data, ignore_index=True)
        # print(df['cau'])
        plot_compare(env_id, df, fign_seaborn, save_fig=True)
