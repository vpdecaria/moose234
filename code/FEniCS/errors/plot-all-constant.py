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
	PressureMax = np.zeros_like(dt)

	for j,line in enumerate(f):
		data = line.strip().split(',')
		data = [float(i) for i in data]
		dt[j] = data[0]
		l2L2[j] = data[1]
		l2H1[j] = data[2]
		l2L2Pressure[j] = data[3]
		PressureMax[j]=data[4]



	f.close()
	return [dt,l2L2,l2H1,l2L2Pressure,PressureMax]

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
	x_u=0.2
	
	#Uncomment for velocity
	#plt.loglog([x_l,x_u],[.5*x_l**2,.5*x_u**2],'k--',label = 'slope 2')
	#Uncomment for pressure
	plt.loglog([x_l,x_u],[.15*x_l**2,.15*x_u**2],'k--',label = 'slope 2')
	#Uncomment for velocity
	#plt.loglog([x_l,x_u],[x_l**3,x_u**3],'k-.',label = 'slope 3')
	x_l=0.00625
	x_u=0.05

	#plt.loglog([x_l,x_u],[x_l**3,x_u**3],'k-.',label = 'slope 3')
	x_l=0.00625
	x_u=0.2
	plt.loglog([x_l,x_u],[3*x_l**4,3*x_u**4],'k:',label = 'slope 4')
	plt.legend(loc = 4)
	
	
	
	
	plt.show()
	

[dt,l2L20,l2H10,l2L20Pressure,PMAX2] = getData('convergenceTestOrder2/order-2.txt')	
[dt3,l2L203,l2H103,l2L20Pressure3,PMAX3] = getData('convergenceTestOrder3/order-3.txt')	
[dt4,l2L204,l2H104,l2L20Pressure4,PMAX4] = getData('convergenceTestOrder4/order-4.txt')


#l2L2
print(dt)
slopel2L20 = stats.linregress(np.log(dt),np.log(l2L20))[0]
slopel2L203 = stats.linregress(np.log(dt3),np.log(l2L203))[0]
slopel2L204 = stats.linregress(np.log(dt4),np.log(l2L204))[0]

xLabel = r'$k$'
xLabel = r'$\Delta t$'
yLabel = r'$\|u-u_h\|_{l2L2}/\|u\|_{l2L2}$'
#title = r'Velocity Error'
title='Nonadaptive Velocity Error'
#lineType = ['k','k--','k-.','k.-']
lineType = []
lineType = ['k','b','g','r.-']
lineType = ['k','b','r','k']
markers = ['v','o','s','^']
legend= ['BDF3-Stab, m = '+'{0:.3f}'.format(slopel2L20),\
		'BDF3, m = '+'{0:.3f}'.format(slopel2L203),\
		'BDF3-4, m = '+'{0:.3f}'.format(slopel2L204)]
legend= ['BDF3-Stab',\
		'BDF3',\
		'FBDF4']
plotCompareMethods(dt,[l2L20,l2L203,l2L204],xLabel,yLabel,title,lineType,legend,markers)


#l2L2 pressure
slopel2L20 = stats.linregress(np.log(dt),np.log(l2L20Pressure))[0]
slopel2L203 = stats.linregress(np.log(dt3),np.log(l2L20Pressure3))[0]
slopel2L204 = stats.linregress(np.log(dt4),np.log(l2L20Pressure4))[0]


xLabel = r'k'
xLabel = r'$\Delta t$'
yLabel = r'$\|p-p_h\|_{l2L2}/\|p\|_{l2L2}$'
#title = r'Pressure Error'
title='Nonadaptive Pressure Error'
lineType = ['k','k--','k-.','k.-']
lineType = ['k','k--','k-.']
lineType = ['k','b','r','k','m-.','c:','g']
markers = ['v','o','s','^','P','*','|']
"""
legend= ['BDF3-Stab, m = '+'{0:.3f}'.format(slopel2L20),\
		'BDF3, m = '+'{0:.3f}'.format(slopel2L203),\
		'BDF3-4, m = '+'{0:.3f}'.format(slopel2L204)]
"""
legend= ['BDF3-Stab',\
		'BDF3',\
		'FBDF4']
plotCompareMethods(dt,[l2L20Pressure,l2L20Pressure3,l2L20Pressure4],xLabel,yLabel,title,lineType,legend,markers)

plotCompareMethods(dt,[PMAX2,PMAX3,PMAX4],xLabel,yLabel,title,lineType,legend,markers)

#PRESSURE lINFL2
#slopePressurelINFL2 = stats.linregress(np.log(dt),np.log(pressurelINFL2))[0]
#slopePressurelINFL2C = stats.linregress(np.log(dt),np.log(pressurelINFL2C))[0]

#print(slopePressurelINFL2C)
#xLabel = 'k'
#title = r'$||p_h - p||_{l^{\infty}L^2}/||p||_{l^{\infty}L^2}$'
#lineType = ['k','k--']
#legend= ['AC, m = '+'{0:.3f}'.format(slopePressurelINFL2), 'Corrected AC, m = '+'{0:.3f}'.format(slopePressurelINFL2C)]
#plotCompareMethods(dt,[pressurelINFL2,pressurelINFL2C],xLabel,title,lineType,legend)


