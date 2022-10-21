import numpy as np
from numpy.random import normal

class Error():
    def __call__(self):
        return 0

class NormalError(Error):

    def __init__(self,sigma):
        self.sigma=sigma

    def __call__(self):
        return normal(0,self.sigma)

class LinearBandit():

    def __init__(self,theta_star,error,arms):
        self.theta_star=theta_star
        self.error=error
        self.arms=arms
        self.best=max([ np.inner(arm,theta_star) for arm in arms])

    def __call__(self,k):
        return np.inner(self.arms[k],self.theta_star)+self.error()


