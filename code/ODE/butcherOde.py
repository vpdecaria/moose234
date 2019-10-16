import numpy as np


def exact(t):
	return np.array([1./2 + np.sqrt((1./4 + 5./35*np.exp(-t)))])


def f(t,y):
	return np.array([y[0]*(1-y[0])/(2*y[0]-1)])
	
def Jf(t,y):
	return np.array([(-2*y[0]**2+2*y[0]-1)/(2*y[0]-1)**2])


T = 5

numerical_data= False