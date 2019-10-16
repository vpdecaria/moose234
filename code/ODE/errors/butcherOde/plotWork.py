import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

#A driver to access simulation results

errorFile = open('allErrors.txt','w')
def getWorkData(filename):

	f = open(filename,'r')

	#First 3 lines are header information
	f.readline()
	f.readline()
	f.readline()

	counter = 0

	while True:
		line = f.readline()
		if line == 'END\n':
			break
		counter += 1
	f.readline()
	line = f.readline()
	data = line.strip().split(',')
	data = [float(i) for i in data]
	tol = data[0]
	l2L2 = data[1]
	l2H1 = data[2]
	l2L2Pressure = data[3]
	rejections = data[4]
	
	
	f.close()
	return [counter,tol,l2L2,l2H1,l2L2Pressure,rejections]

#Plotting Method

def compileErrors(fName,lowerTol,UpperTol):
	counter = np.array([])
	tol = np.array([])
	l2L2 = np.array([])
	l2H1 = np.array([])
	l2L2Pressure = np.array([])
	rejections = np.array([])
	for i in range(UpperTol,lowerTol-1,-1):
		[countertemp,toltemp,l2L2temp,l2H1temp,l2L2Pressuretemp,rejectionstemp] = getWorkData('1e'+str(i)+fName)
		#output = '%1.E& %1.2E& %1.2E& %1.2E& %i& %i' % (toltemp,l2L2temp,l2H1temp,l2L2Pressuretemp,rejectionstemp,countertemp)
		output = '%1.E& %1.2E& %1.2E& %i& %i' % (toltemp,l2L2temp,l2L2Pressuretemp,rejectionstemp,countertemp)
		
		if(i != UpperTol):
			logRatio = np.log(np.float64(counterOld)/countertemp)
			l2Converge = -np.log(l2L2old/l2L2temp)/logRatio
			H1Converge = -np.log(l2H1old/l2H1temp)/logRatio
			pressureConverge = -np.log(l2L2Pressureold/l2L2Pressuretemp)/logRatio
			#additionalOut = ' &%1.2f & %1.2f& %1.2f' % (l2Converge,H1Converge,pressureConverge)
			additionalOut = ' &%1.2f & %1.2f' % (l2Converge,pressureConverge)
			output += str(additionalOut)
		output = str(output).replace('E-0','E-') + '\\\\\n'
		errorFile.write(output)
		counter = np.append(counter,countertemp)
		tol = np.append(tol,toltemp)
		l2L2 = np.append(l2L2,l2L2temp)
		l2H1 = np.append(l2H1,l2H1temp)
		l2L2Pressure = np.append(l2L2Pressure,l2L2Pressuretemp)
		rejections = np.append(rejections,rejectionstemp)
		
		l2L2old = l2L2temp
		l2H1old = l2H1temp
		l2L2Pressureold = l2L2Pressuretemp
		counterOld = countertemp
		
	return [counter,tol,l2L2,l2H1,l2L2Pressure,rejections]
		
#t is the independent variable.
#methodData is a list of data from however many methods. Each data should be the same size as dt.
#title - string, title of plot
#xLabel - string, x-axis label
#lineType - list of strings to indicate type of lines we want drawn.
#labels - list of strings for titles in legend
def plotCompareMethods(dt, methodData,xLabel,title,lineType,labels):
	numberOfMethods = len(methodData)  #How many methods are we comparing?
	if(lineType == []):
		for j, data in enumerate(methodData):
			plt.loglog(dt,data,label = labels[j])
	else:
		for j, data in enumerate(methodData):
			plt.loglog(dt,data,lineType[j],label = labels[j])
	plt.xlabel(xLabel,fontsize = 18)
	plt.title(title,fontsize = 18)
	plt.xlim([25e-4,dt[0]+0.02])
	plt.legend(loc = 4)
	
	plt.show()

#gamma  = 1e0
#[counter,tol,l2L2,l2H1,l2L2Pressure,rejections] = compileErrors('second-constant-step.txt',-7,-1)
#work = counter + rejections
#plt.loglog(work,l2L2,'k:',marker = '+',label = '2nd order')
lower_tol = -10

[counter,tol,l2L2,l2H1,l2L2Pressure,rejections] = compileErrors('-order-2.txt',lower_tol,-1)
work = counter + rejections
plt.loglog(work,l2L2,'r:',marker = 'v',label = '2')
[counter,tol,l2L2,l2H1,l2L2Pressure,rejections] = compileErrors('-order-3.txt',lower_tol,-1)
work = counter + rejections
plt.loglog(work,l2L2,'b--',marker = 'o', label ='3')
[counter,tol,l2L2,l2H1,l2L2Pressure,rejections] = compileErrors('-order-23.txt',lower_tol,-1)
work = counter + rejections
plt.loglog(work,l2L2,'g-.',marker = 's',label ='23')

[counter,tol,l2L2,l2H1,l2L2Pressure,rejections] = compileErrors('-order-4.txt',lower_tol,-1)
work = counter + rejections
plt.loglog(work,l2L2,'k-.',marker = 'D',label ='4')

[counter,tol,l2L2,l2H1,l2L2Pressure,rejections] = compileErrors('-order-34.txt',lower_tol,-1)
work = counter + rejections
plt.loglog(work,l2L2,'m-.',marker = 'H',label ='34')

[counter,tol,l2L2,l2H1,l2L2Pressure,rejections] = compileErrors('-order-234.txt',lower_tol,-1)
work = counter + rejections
plt.loglog(work,l2L2,'w-.',marker = '>',label ='234')

[counter,tol,l2L2,l2H1,l2L2Pressure,rejections] = compileErrors('-order-24.txt',lower_tol,-1)
work = counter + rejections
plt.loglog(work,l2L2,'y-.',marker = '<',label ='24')

plt.xlabel("Steps Taken + Rejections")
plt.ylabel("Velocity Error")
plt.legend()
plt.title("Error")
plt.show()

exit()


[counter,tol,l2L2,l2H1,l2L2Pressure,rejections] = compileErrors('first.txt',-6,-1)
work = counter + rejections
plt.loglog(counter,l2L2,'r:',marker = 'o',label = 'VSVOBE1')
[counter,tol,l2L2,l2H1,l2L2Pressure,rejections] = compileErrors('second.txt',-7,-1)
work = counter + rejections
plt.loglog(counter,l2L2,'b--',marker = 'o', label ='VSVOBE2')
[counter,tol,l2L2,l2H1,l2L2Pressure,rejections] = compileErrors('third.txt',-6,-1)
work = counter + rejections
plt.loglog(counter,l2L2,'g-.',marker = 's',label ='VSVOBE3')
[counter,tol,l2L2,l2H1,l2L2Pressure,rejections] = compileErrors('fourth.txt',-5,-1)
print([counter,tol,l2L2,l2H1,l2L2Pressure,rejections])
work = counter + rejections
plt.loglog(counter,l2L2,'k',marker = '^',label ='VSVOBE4')
plt.xlabel("System Solves")
plt.ylabel("Velocity Error")
plt.legend()
plt.show()

"""
plt.figure(1)
plt.subplot(411)

plt.semilogy(t,k)
plt.semilogy(t1,k1)

plt.subplot(412)
plt.plot(t,filters)
plt.plot(t1,filters1)
#
#plt.plot(t,TotalModelDissipation1e6)
#plt.plot(t,KineticEnergy1e6+ TotalModelDissipation1e6)
#plt.plot(t,TotalModelDissipation1e10)
#plt.plot(t,TotalModelDissipation1e2)
plt.subplot(413)
tFine = np.linspace(0,10,10000)
#plt.show()
plt.plot(t,normU)
plt.plot(t1,normU1)
#plt.plot(t,normUExact)
exact = [np.abs((np.sqrt(2)*np.pi)*(np.cos(T) - np.cos(np.pi*T))) for T in tFine]
plt.plot(tFine,exact)
plt.subplot(414)
plt.semilogy(t,Uerror)
plt.semilogy(t1,Uerror1)
plt.show()
"""
"""
time = 78
gammas = [1.0,1e2,1e4,1e6,1e8,1e10]
ModelDissipation1sec = [ModelDissipation1e0[time], ModelDissipation1e2[time],\
	ModelDissipation1e4[time],ModelDissipation1e6[time],\
	ModelDissipation1e8[time],ModelDissipation1e10[time]]
	
TotalModelDissipation1sec = [TotalModelDissipation1e0[time], TotalModelDissipation1e2[time],\
	TotalModelDissipation1e4[time],TotalModelDissipation1e6[time],\
	TotalModelDissipation1e8[time],TotalModelDissipation1e10[time]]

plt.semilogx(gammas,TotalModelDissipation1sec)
plt.title("Cumulative Dissipation at time " + str(t[time]), fontsize = 18)
plt.xlabel(r"\gamma")
plt.show()
"""
