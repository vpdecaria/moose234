import numpy as np
#from scipy import linalg


def first_difference(T,Y):
	"""
	T is a vector of two times, greatest to least
	Y is a vector older differences, [dj,d(j+1)]
	See algorthm in paper
	"""
	return (Y[0]-Y[1])/(T[0] - T[1])

def backward_differences(T):
	"""
	Generate the divided difference coefficients. T is a vector of 
	times from least to greatest
	T = [t_n,t_n+1,....,t_m+m]
	"""
	numOfTimes = len(T)
	#the number of steps in the method
	m = numOfTimes - 1
	#generate the initial conditions for the differences, which
	#is just the standard basis.
	D = np.array([ [np.float64((i+1)==(numOfTimes-j)) for i in xrange(numOfTimes)] for j in xrange(numOfTimes)])
	differences = np.zeros_like(D)
	differences[0] = D[0]
	
	
	for q in xrange(1,numOfTimes):
		for j in xrange(numOfTimes - q):
			D[j] = first_difference([T[m-j],T[m-j-q]],[D[j],D[j+1]])
			differences[q] = D[0]
	return differences
	
def bdf_coefficients(T):
	differences = backward_differences(T)
	m = len(T)-1
	return np.sum(np.prod([T[m]-T[m-i] for i in xrange(1,j)])*differences[j] for j in xrange(1,m+1))

print(backward_differences([1,2,3,4,5]))

#differences([1.0,2.0,3.0,4.5])

backward_differences([1.0,2.0,3.0,4.0,5.5])
print(bdf_coefficients([1,2]))

print(bdf_coefficients([1,2,3.1]))
print(bdf_coefficients([1,2,3,4]))
