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
def plotCompareMethods(dt, methodData,xLabel,title,lineType,labels,markers):
	numberOfMethods = len(methodData)  #How many methods are we comparing?
	if(lineType == []):
		for j, data in enumerate(methodData):
			plt.loglog(dt,data,label = labels[j])
	else:
		for j, data in enumerate(methodData):
			plt.loglog(dt,data,lineType[j],label = labels[j],marker = markers[j])
	plt.xlabel(xLabel,fontsize = 18)
	plt.title(title,fontsize = 18)
	#plt.xlim([50e-4,dt[0]+0.02])
	#plt.ylim([1e-6,1.7e-1])
	plt.legend(loc = 4)
	
	plt.show()
	
#bdf2
[dt,l2L20,l2H10,l2L20Pressure] = getData('order-1.txt')	
	
#bdf3-2
[dt,l2L21,l2H11,l2L21Pressure]  = getData('order-2.txt')	

#bdf3
[dt,l2L22,l2H12,l2L22Pressure]  = getData('order-3.txt')	

#bdf3-4
[dt,l2L23,l2H13,l2L23Pressure]  = getData('order-4.txt')

#bdf4	
[dt,l2L24,l2H14,l2L24Pressure]  = getData('order-5.txt')

#l2L2
print(dt)
slopel2L20 = stats.linregress(np.log(dt),np.log(l2L20))[0]
slopel2L21 = stats.linregress(np.log(dt),np.log(l2L21))[0]
slopel2L22 = stats.linregress(np.log(dt),np.log(l2L22))[0]
slopel2L23 = stats.linregress(np.log(dt),np.log(l2L23))[0]
slopel2L24 = stats.linregress(np.log(dt),np.log(l2L24))[0]

#print(slopel2L22)
xLabel = 'k'
title = r'Error'
#lineType = ['k','k--','k-.','k.-']
lineType = []
lineType = ['k','b--','g-.','r.-']
lineType = ['r:','b--','g-.','k','c']
markers = ['v','o','s','^','+']
legend= ['BDF2, m = '+'{0:.3f}'.format(slopel2L20),'BDF3-2, m = '+'{0:.3f}'.format(slopel2L21),\
	'BDF3, m = '+'{0:.3f}'.format(slopel2L22),'BDF3-4, m = '+'{0:.3f}'.format(slopel2L23),'BDF4, m = '+'{0:.3f}'.format(slopel2L24)]
#plotCompareMethods(dt,[l2L20,l2L21],xLabel,title,lineType,legend,markers)
plotCompareMethods(dt,[l2L20,l2L21,l2L22,l2L23,l2L24],xLabel,title,lineType,legend,markers)
