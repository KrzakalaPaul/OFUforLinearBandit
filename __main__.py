import numpy as np


from linear_bandit import LinearBandit,NormalError
from ofu_agent import OFUAgent
from plot_regret import setup_regret,plot_regret
from plot_confidence_set import setup_confidence,plot_conficence_set

import argparse


parser = argparse.ArgumentParser()

parser.add_argument("-c", "--custom", type=int, choices=[0, 1, 2],
                    help="enable customisation of the parameters")

args = parser.parse_args()
custom=args.custom


### ------------------- PLOTTING REGRET : ----------------------- ## 

default_setup=setup_regret(custom)

error= NormalError(default_setup['sigma'])
linearbandit=LinearBandit(default_setup['theta_start'],error,default_setup['arms'])

policy=OFUAgent(default_setup['lambda'],default_setup['R'],default_setup['S'],default_setup['delta'],default_setup['arms'])
plot_regret(linearbandit,policy,horizon=1000,n_trajectories=10)


### ------------------- PLOTTING THE CONFIDENCE SET (2D ONLY) : ----------------------- ## 

"""
default_setup=setup_confidence(custom)

error= NormalError(default_setup['sigma'])
linearbandit=LinearBandit(default_setup['theta_start'],error,default_setup['arms'])

policy=OFUAgent(default_setup['lambda'],default_setup['R'],default_setup['S'],default_setup['delta'],default_setup['arms'])

plot_conficence_set(linearbandit,policy,horizon=20,refresh=1)
"""