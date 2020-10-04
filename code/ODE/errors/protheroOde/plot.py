import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

#A driver to access simulation results
def getData(filename):

	f = open(filename,'r')

	#First 3 lines are header information
	f.readline()
	f.readline()
	f.readline()
	
	dt = np.zeros(6)
	
	l2L2 = np.zeros_like(dt)
	l2H1 = np.zeros_like(dt)
	l2L2Pressure = np.zeros_like(dt)

	for j,line in enumerate(f):
		data = line.strip().split(',')
		data = [float(i) for i in data]
		dt[j] = data[0]
		l2L2[j] = data[1]
		l2H1[j] = data[2]
		l2L2Pressure[j] = data[3]


	f.close()
	return [dt,l2L2,l2H1,l2L2Pressure]

#Plotting Method

#t is the independent variable.
#methodData is a list of data from however many methods. Each data should be the same size as dt.
#title - string, title of plot
#xLabel - string, x-axis label
#lineType - list of strings to indicate type of lines we want drawn.
#labels - list of strings for titles in legend
def plotCompareMethods(dt, methodData,xLabel,yLabel,title,lineType,labels,markers):
	numberOfMethods = len(methodData)  #How many methods are we comparing?
	if(lineType == []):
		for j, data in enumerate(methodData):
			plt.loglog(dt,data,label = labels[j])
	else:
		for j, data in enumerate(methodData):
			plt.loglog(dt,data,lineType[j],label = labels[j],marker = markers[j])
	plt.xlabel(xLabel,fontsize = 18)
	plt.ylabel(yLabel,fontsize=18)
	plt.title(title,fontsize = 18)

	x_l = .00625
	x_u = .2
	c= 2e-1
	plt.loglog([x_l, x_u], [c*x_l**3, c*x_u**3], 'k--', label = 'slope = 3')

	x_l = .00625
	x_u = .2
	c= 3e-1
	plt.loglog([x_l, x_u], [c*x_l**4, c*x_u**4], 'k:', label = 'slope = 4')
	#plt.loglog()
	#plt.xlim([50e-4,dt[0]+0.02])
	#plt.ylim([1e-6,1.7e-1])
	plt.legend(loc = 4)
	plt.tight_layout()
	plt.show()
	
#Backward Euler
[dt,l2L22,l2H10,l2L20Pressure] = getData('order-2.txt')	
[dt,l2L23,l2H10,l2L20Pressure] = getData('order-3.txt')	
[dt,l2L24,l2H10,l2L20Pressure] = getData('order-4.txt')	

xLabel = 'k'
yLabel = r'$\ell ^2 $ error '
title = r'Error for Prothero and Robinson test problem,'+'\n'+ r'$\lambda = -1e6$'
#lineType = ['k','k--','k-.','k.-']
lineType = []
lineType = ['k','b--','g-.','r.-']
lineType = ['r','b','g-.','k']
markers = ['v','o','s','^']
legend= ['BDF3-Stab','FBDF4']
plotCompareMethods(dt,[l2L22,l2L24],xLabel,yLabel,title,lineType,legend,markers)

