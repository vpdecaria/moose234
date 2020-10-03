import numpy as np

lam = -1e6
def exact(t):
	return np.array([np.sin(t)])


def f(t,y):
	return np.array([np.cos(t) + lam*(y[0] - exact(t)[0]) ])
	
def Jf(t,y):
	return np.array([lam])

#lam = 0

T = 2*np.pi

numerical_data= False