import numpy as np
from scipy import linalg


def F(x):
	return np.array([x[0]**2 - 2])
	
def J(x):
	return np.array([2*x[0]])

def newton(guess,F,J,tol,maxIter):

	old_guess = guess
	for i in range(maxIter):
		Jac= J(guess)
		guess = linalg.solve(Jac,Jac.dot(guess)-F(guess))
		if(linalg.norm(guess - old_guess) < linalg.norm(old_guess)*tol):
			return guess
		old_guess = guess
	exit("maximum iterations reached")

