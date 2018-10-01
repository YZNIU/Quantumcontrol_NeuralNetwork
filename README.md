# Quantum Control in the Eyes of Neural Network

 
Qc_env.py is the customerized environment used inside gym/env/classic_control for optimizing pulse sequence of single-qubit gate simulation. 


# reference:

1.https://arxiv.org/abs/1506.02438
2.http://proceedings.mlr.press/v37/schulman15.pdf


# Installation

0. Clone openai repo, e.g. git clone https://github.com/openai/gym.git
1. Clone trpo repo, e.g. git clone https://github.com/joschu/modular_rl.git

2. Install everything according to openai gym.

        Installing everything

        To install the full set of environments, you'll need to have some system packages installed. We'll build out the list here over time; please let us know what you end up installing on your platform.

        On OSX:

        brew install cmake boost boost-python sdl2 swig wget
        
        On Ubuntu 14.04:

        apt-get install -y python-numpy python-dev cmake zlib1g-dev libjpeg-dev xvfb libav-tools xorg-dev python-opengl libboost-all-dev libsdl2-dev swig
       
        Once you're ready to install everything, run pip install -e '.[all]' (or pip install 'gym[all]').


3.  add qc_env.py into the folder /openai/gym/gym/envs/classical_control/
4.  add this line:

```
from gym.envs.classic_control.qc_env import QCEnv
```

into `__init__.py` under /openai/gym/gym/envs/classical_control/

5.  add following lines: 

```
    register(
        id='QCEnv-v0',
        entry_point='gym.envs.classic_control:QCEnv',
        max_episode_steps=1000,
    )
```

into `__init__.py` under `/openai/gym/gym/envs/`

# Run 

1. enter the openai/modular_rl/folder

2. from the command line, e.g.
  
