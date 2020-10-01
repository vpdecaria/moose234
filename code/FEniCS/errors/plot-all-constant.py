import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

#A driver to access simulation results

markers = ['v','o','s','^','+']

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
	VelLast = np.zeros_like(dt)
	PreLast = np.zeros_like(dt)

	for j,line in enumerate(f):
		data = line.strip().split(',')
		data = [float(i) for i in data]
		dt[j] = data[0]
		l2L2[j] = data[1]
		l2H1[j] = data[2]
		l2L2Pressure[j] = data[3]
		VelLast[j]=data[4]
		PreLast[j]=data[5]



	f.close()
	return [dt,l2L2,l2H1,l2L2Pressure,VelLast,PreLast]

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
	x_l=0.00625
	x_u=x_l * 4

	vel = False
	c = .5 if vel else .8;

	
	#Uncomment for velocity
	#plt.loglog([x_l,x_u],[.5*x_l**2,.5*x_u**2],'k--',label = 'slope 2')
	#Uncomment for pressure
	plt.loglog([x_l,x_u],[c*x_l**2,c*x_u**2],'k:')
	#Uncomment for velocity
	x_l=0.00625
	x_u=x_l * 4

	c = 2 if vel else 3;
	plt.loglog([x_l,x_u],[c*x_l**3,c*x_u**3],'k--')


	c = 5 if vel else 15;

	#plt.loglog([x_l,x_u],[x_l**3,x_u**3],'k-.',label = 'slope 3')

	plt.loglog([x_l,x_u],[c*x_l**4,c*x_u**4],'k-.')
	plt.legend(loc = 4,fontsize = 14)
	
	
	
	
	plt.show()
	

[dt,l2L20,l2H10,l2L20Pressure,VLAST2,PLAST2] = getData('convergenceTestOrder2/order-2.txt')	
[dt3,l2L203,l2H103,l2L20Pressure3,VLAST3,PLAST3] = getData('convergenceTestOrder3/order-3.txt')	
[dt4,l2L204,l2H104,l2L20Pressure4,VLAST4,PLAST4] = getData('convergenceTestOrder4/order-4.txt')

xLabel = r'$\Delta t$'
yLabel = 'final relative error'
#title = r'Velocity Error'
title='Nonadaptive Velocity Error'

#plotCompareMethods(dt,[l2L20,l2L203,l2L204],xLabel,yLabel,title,lineType,legend,markers)

lineType = ['k','b','r','k','m-.','c:','g']
markers = ['v','o','s','^','P','*','|']

legend= ['BDF3-Stab',\
		'BDF3',\
		'FBDF4']
#plotCompareMethods(dt,[l2L20Pressure,l2L20Pressure3,l2L20Pressure4],xLabel,yLabel,title,lineType,legend,markers)
title='Nonadaptive Velocity Error'
#plotCompareMethods(dt,[VLAST2,VLAST3,VLAST4],xLabel,yLabel,title,lineType,legend,markers)
title='Nonadaptive Pressure Error'
plotCompareMethods(dt,[PLAST2,PLAST3,PLAST4],xLabel,yLabel,title,lineType,legend,markers)


