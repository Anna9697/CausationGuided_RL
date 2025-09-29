# CausationGuided-RL
## Control Synthesis of Cyber-Physical Systems for Real-Time Specifications through Causation-Guided Reinforcement Learning  

This is a tool to synthesize controllers of Cyber-Physical Systems for Real-Time Specifications through Causation-Guided Reinforcement Learning. 
It is developed using the stable-baselines3 framework (https://github.com/DLR-RM/stable-baselines3) and a C++ lib no the top of Breach online monitoring tool (https://github.com/decyphir/breach).
The generated C++ dynamic link library already exists in the current directory (. so files)

## 1. Prerequisites

Use Ubuntu 20.04 and above.  
Packages: libboost-all-dev, python-dev, python-pip, antlr4   

Install mujoco (https://github.com/openai/mujoco-py).   

Python packages    
<Pkg>        <Preferable Version>   
Python         3.7     
torch          1.11.0      
gym            0.15.3     
PyOpenGL       3.1.5    
glfw           2.4.0   
imageio        2.10.3   
mujoco-py      >=2.1.2   
safty-gym      0.0.0

The above version are highly recommended due issues with other versions.   
For instance, mujoco==2.1.2.12 conflicts with gym 0.24.0.   

## 2. Installation
Create a python3.7 virtual environment and do the following:   

Unzip the file .zip and install the package:    
```
cd Causation-RL/
pip install -e .
```

After installation, add the path to the "CausationGuided_RL" package to the PYTHONPATH variable in ~/.bashrc file.    
For example, if the CausationGuided_RL package is ta location /home/PC/CausationGuided_RL/, then add the  
following lines in the ~/.bashrc file:  
export PYTHONPATH=/home/PC/CausationGuided_RL/:$PYTHONPATH

Alternatively, instead to adding these lines to ~/.bashrc, you can run these two lines in the terminal but it will be valid for a session only.   


(In case this variable is not set properly, you might notice error saying  
"TypeError: learn() got an unexpected keyword argument 'reward_type').   



## 3. Running Experiments  

### 3.1 Run the training program    

```
cd src/
```

```
python run_experiments.py --env=<Env> --reward=<reward-id> --sem=<semantics-id> --run=<run-id>
```
- where Env = {CartPole-v1, PointGoal1-v0, Hopper-v3, Walker2d-v3}   
- reward-id = {0, 1}, denote using stl or not
- methods with semantics-id:
  - BAS: --reward=0 --sem=none
  - CLS: --reward=1 --sem=cls
  - LSE: --reward=1 --sem=lse
  - SSS: --reward=1 --sem=sss
  - CAU: --reward=1 --sem=cau_app
- run-id is a unique integer to be provided by the user. This purpose is to distinguish one set of experiments from the other.

Note that the training is performed 10 times by default (on different seeds). 

Once a training finishes, it will create a controller file named `xxx_Env_reward-id_semantics-
id_run-id.zip` is stored in `./src/result` , and the training training process data is stored in
`./src/log_stl_<sem-id>`.

For example, train with CAU method in the CartPole benchmark, run the command: 
```
python run_experiments.py --env=CartPole-v1 --reward=1 --sem=cau_app --run=0
```

### 3.2 Run the evaluation program  

To evaluation of a controller for environment <Env> and different method, run the command:

```
python evaluator.py --model=<model-id> --env=<Env> --stl=<reward-name> --sem=<semantics-
id> --seed=<run-id> --tau=<k>
```
where
- reward-name is 'normal' or 'stl'.
- model-id is 'ppo' or 'sac'.
- k is an intiger, if k=0 (default), the parameter k is set to the minimum value according to
Equation (7).

For example, to evaluate the controller for Cart-Pole, run the command: 
```
python evaluator.py --model=ppo --env=CartPole-v1 --stl=stl --sem=cau_app --seed=0
```
then it will test the controller from seed 0 to 9. The evaluation result will be printed in the terminal
after the run is completed.

### Files
All the source code is inside the ```src/``` folder.


### Reproducibilty
All experiments are performed on an Ubuntu 22.04 machine equipped with an Intel Core i7-2700F CPU, an NVIDIA GeForce RTX 3080 GPU and 32GB of RAM. 

Completely reproducible results are not guaranteed across PyTorch releases or different platforms.   
Refer to the following notes by   
PyTorch (https://pytorch.org/docs/stable/notes/randomness.html) and   
stable-baselines (https://stable-baselines3.readthedocs.io/en/master/guide/algos.html#reproducibility)    
 
