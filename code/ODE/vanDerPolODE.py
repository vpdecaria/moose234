import numpy as np

mu = 1000

def exact(t):
	return np.array([[-2],[-134]])


def f(t,y):
	return np.array([[y[1][0]],[mu*(1-y[0][0]**2)*y[1][0] - y[0][0]]])
	
def Jf(t,y):
	return np.array([[0,1],[-2*mu*y[0][0]*y[1][0] -1,mu*(1-y[0][0]**2)]])


T = 3000

numerical_data=True


t_data_file = 't-van-der-pol.txt'
y_data_file = 'y-van-der-pol.txt'
