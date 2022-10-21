from cProfile import label
from numpy.random import normal
from numpy.linalg import norm
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib import cm
import matplotlib.animation as animation

def setup_confidence(custom):
    dim=2

    if custom==True:
        print('Parameters for the confidence interval animation ?')
        n_arms=int(input('Number Of Arms'))
        sigma=float(input('sigma noise'))
        lbda=float(input('lambda'))
        delta=float(input('delta'))

    else:
        n_arms=15
        sigma=3
        lbda=2
        delta=0.05

    dic={}
    dic['dim']=dim
    dic['n_arms']=n_arms
    dic['sigma']=sigma
    dic['lambda']=lbda
    dic['delta']=delta

    dic['theta_start']=np.array([5,5])
    dic['arms']=[normal(0,1,dim) for _ in range(n_arms)]
    dic['sigma']=sigma

    dic['R']=dic['sigma']
    dic['S']=norm(dic['theta_start']) 
    dic['horizon']=20
    dic['refresh']=1

    return dic


def plot_conficence_set(bandit,policy,horizon=100,refresh=10):  

    viridis = cm.get_cmap('plasma', horizon)

    theta_star=bandit.theta_star

    nb_points = 100

    u = np.linspace(-15, 15, nb_points)

    x, y = np.meshgrid(u, u)

    fig, ax = plt.subplots()
    fig.tight_layout()
    plt.xlim(-15, 15)
    plt.ylim(-15, 15)
    ax=plt.gca()
    ax.set_aspect('equal', adjustable='box')

    plt.scatter([theta_star[0]],[theta_star[1]],color='r', marker='*',label=r'$\theta^*$')
    plt.title("Confidence Set at each step t")
    plt.legend()

    for k in range(horizon):

        b=policy.b()
        theta_hat=policy.theta_hat()
        Sigma=policy.V_t

        if k%refresh==0:

            X = np.vstack([x.flatten()-theta_hat[0], y.flatten()-theta_hat[1]])

            f_x = np.sqrt(np.abs(np.dot(np.dot(X.T, Sigma), X)))
            f_x = np.diag(f_x).reshape(nb_points, nb_points)

            
            plt.contour(x, y, f_x, [b],colors=[viridis(k/horizon)])
            plt.pause(0.01)

        arm=policy.pick_arm()
        reward=bandit(arm)
        policy.update(arm,reward)

        

    
    plt.show()
    