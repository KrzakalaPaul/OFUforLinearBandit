# OFU for LinearBandit
 Application of the OFU principle to linear bandit using the upper confidence bound derived from https://arxiv.org/abs/1102.2670.
 
 ## Setting :
 
 We focus on the case where there is a finite number of arms $x_1,...,x_K$. In that case we can use corollary 8 to derive a simple algorithm, see [section 3]( https://drive.google.com/file/d/1A4grRyAHupf3nybDi-LsxqwKadyXD4VY/view?usp=sharing).


 ## The code :
 
 Running main.py launches two experiments: 
 
 1. In high dimension (by default d=10) with a plot of the regret for different episodes and the other 
 2. In two dimension with a plot of the evolution of the confidence set. 
 
 Adding the flag parameter --custom 1 will enable to tune the hyperparameters and the setting (number of arms, dimensions etc...)
 
 
