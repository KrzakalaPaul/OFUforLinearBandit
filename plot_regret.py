import numpy as np
from numpy.random import normal
from numpy.linalg import norm
import matplotlib.pyplot as plt 
from ofu_agent import weighted_norm




def setup_regret(custom):

    if custom==True:
        print('Parameters for the regret plot ?')
        dim=int(input('Dimension'))
        n_arms=int(input('Number Of Arms'))
        sigma=float(input('sigma noise'))
        lbda=float(input('lambda'))
        delta=float(input('delta'))

    else:
        dim=10
        n_arms=20
        sigma=0.1
        lbda=1
        delta=0.05

        
    dic={}
    dic['dim']=dim
    dic['n_arms']=n_arms
    dic['sigma']=sigma
    dic['lambda']=lbda
    dic['delta']=delta

    dic['theta_start']=np.zeros(dim)+1
    dic['arms']=[normal(0,1,dim) for _ in range(n_arms)]
    dic['sigma']=0.1

    dic['R']=dic['sigma']
    dic['S']=norm(dic['theta_start']) 

    return dic




def plot_regret(bandit,policy,horizon=100,n_trajectories=1):

    fig,ax=plt.subplots()
    fig.tight_layout()
    plt.title('Regret for differents episodes')
    plt.xlabel('Regret')
    plt.ylabel('number of iterations t')
    
    
    for _ in range(n_trajectories):

        policy.reset()
        Regret=0
        Regret_plot=np.zeros(horizon)
        inequality_verified=True

        for k in range(horizon):

            arm=policy.pick_arm()
            reward=bandit(arm)
            policy.update(arm,reward)

            Regret+=bandit.best-reward
            Regret_plot[k]=Regret

            #if inequality_verified:
            #    inequality_verified=weighted_norm(policy.theta_hat()-bandit.theta_star,policy.V_t)<policy.b()

        #if inequality_verified:
        #   plt.plot(Regret_plot,color='green')
        #else:
        #   plt.plot(Regret_plot,color='red')
        plt.plot(Regret_plot)

        
    plt.show()