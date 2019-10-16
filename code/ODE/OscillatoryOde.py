import numpy as np

lam = 1j

def exact(t):
	return np.array([np.exp(lam*t)])


def f(t,y):
	return np.array([lam*y[0]])
	
def Jf(t,y):
	return np.array([lam])


T = 100
