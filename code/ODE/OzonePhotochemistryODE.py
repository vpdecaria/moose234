import numpy as np

#Length of day in seconds
td=86400

k2=1e5
k3=1e-16

def exact(t):
	return np.array([[0],[0],[5e11],[8e11]])


def f(t,y):
	k1= 1e-2*np.max([0,np.sin(2*np.pi*t/td)])
	return np.array([[k1*y[2][0] - k2*y[0][0]],\
		[k1*y[2][0]- k3*y[1][0]*y[3][0]],\
		[k3*y[1][0]*y[3][0] - k1*y[2][0]],\
		[k2*y[0][0] - k3*y[1][0]*y[3][0]]   ])
	
def Jf(t,y):
	k1= 1e-2*np.max([0,np.sin(2*np.pi*t/td)])
	return np.array([[-k2,0,k1,0],\
		[0,-k3*y[3][0],k1,-k3*y[1][0]],\
		[0,k3*y[3][0],-k1,k3*y[1][0]],\
		[k2,-k3*y[3][0],0,-k3*y[1][0]]])


T = 2*td

numerical_data=True


t_data_file = 't-ozone.txt'
y_data_file = 'y-ozone.txt'
