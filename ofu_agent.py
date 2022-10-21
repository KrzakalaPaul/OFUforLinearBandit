import numpy as np
from numpy.random import choice,binomial,uniform,normal
from numpy.linalg import norm
from numpy.linalg import inv
import matplotlib.pyplot as plt 


def chose_max_at_random(estimates):
    return choice(np.argwhere(estimates == np.amax(estimates)).flatten())

def weighted_norm(x,Sigma):
    return np.sqrt(np.inner(x,Sigma@x))

class OFUAgent():

    def __init__(self,lbda,R,S,delta,arms):

        self.arms=arms
        self.L=max([norm(arm,ord=2) for arm in arms])
        self.delta=delta
        self.lbda=lbda
        self.R=R
        self.S=S
        self.d=len(arms[0])
        
        self.reset()

    def reset(self):
        self.t=0
        self.X_tY_t=np.zeros(self.d)
        self.V_t=self.lbda*np.eye(self.d)
        # Improvement : Store V_t^-1 + Sherman–Morrison formula
    
    def update(self,arm,reward):

        arm=self.arms[arm]

        self.X_tY_t+=reward*arm
        self.V_t+=np.outer(arm,arm)
        self.t+=1

    def theta_hat(self):
        return inv(self.V_t)@self.X_tY_t 

    def b(self):

        L=self.L
        d=self.d
        R=self.R
        t=self.t
        S=self.S
        delta=self.delta
        lbda=self.lbda

        return R*np.sqrt(d*np.log((1+t*L/lbda)/delta))+np.sqrt(lbda)*S
    

    def pick_arm(self):
        
        b=self.b()
        theta_hat=self.theta_hat()
        UCB_arm=np.zeros(len(self.arms))

        for k,arm in enumerate(self.arms):
            UCB_arm[k]=( b*weighted_norm(arm,inv(self.V_t))  + np.inner(arm,theta_hat))

        return chose_max_at_random(UCB_arm)