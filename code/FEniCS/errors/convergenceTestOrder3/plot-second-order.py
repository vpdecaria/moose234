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
	#plt.xlim([50e-4,dt[0]+0.02])
	#plt.ylim([1e-6,1.7e-1])
	plt.legend(loc = 4)
	
	plt.show()
	
#Crank-Nicholson
[dt,l2L20,l2H10,l2L20Pressure] = getData('order-1.txt')	




#l2L2
print(dt)
slopel2L20 = stats.linregress(np.log(dt),np.log(l2L20))[0]


xLabel = 'k'
yLabel = r'$\|u-u_h\|_{l2L2}/\|u\|_{l2L2}$'
title = r'Velocity Error'
#lineType = ['k','k--','k-.','k.-']
lineType = []
lineType = ['k','b--','g-.','r.-']
lineType = ['r','b','g-.','k']
markers = ['v','o','s','^']
legend= ['CN, m = '+'{0:.3f}'.format(slopel2L20)]
plotCompareMethods(dt,[l2L20],xLabel,yLabel,title,lineType,legend,markers)


#l2L2 pressure
slopel2L20 = stats.linregress(np.log(dt),np.log(l2L20Pressure))[0]




xLabel = 'k'
yLabel = r'$\|p-p_h\|_{l2L2}/\|p\|_{l2L2}$'
title = r'Pressure Error'
lineType = ['k','k--','k-.','k.-']
lineType = ['k','k--','k-.']
lineType = ['r','b','g','k','m-.','c:','g']
markers = ['v','o','s','^','P','*','|']
legend= ['CN, m = '+'{0:.3f}'.format(slopel2L20)]
plotCompareMethods(dt,[l2L20Pressure],xLabel,yLabel,title,lineType,legend,markers)



#PRESSURE lINFL2
#slopePressurelINFL2 = stats.linregress(np.log(dt),np.log(pressurelINFL2))[0]
#slopePressurelINFL2C = stats.linregress(np.log(dt),np.log(pressurelINFL2C))[0]

#print(slopePressurelINFL2C)
#xLabel = 'k'
#title = r'$||p_h - p||_{l^{\infty}L^2}/||p||_{l^{\infty}L^2}$'
#lineType = ['k','k--']
#legend= ['AC, m = '+'{0:.3f}'.format(slopePressurelINFL2), 'Corrected AC, m = '+'{0:.3f}'.format(slopePressurelINFL2C)]
#plotCompareMethods(dt,[pressurelINFL2,pressurelINFL2C],xLabel,title,lineType,legend)


