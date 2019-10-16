import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

#A driver to access simulation results
def getWorkData(filename):

	f = open(filename,'r')

	#First 3 lines are header information
	f.readline()
	f.readline()
	f.readline()

	counter = 0
	max_pressure_error=-1.0
	while True:
		line = f.readline()
		if line == 'END\n':
			break
		data = line.strip().split(',')
		pressure_error=float(data[8])
		if pressure_error > max_pressure_error:
			max_pressure_error=pressure_error
#			print(max_pressure_error)
		counter += 1
	f.readline()
	line = f.readline()
	data = line.strip().split(',')
	data = [float(i) for i in data]
	l2L2 = data[1]
	l2H1 = data[2]
	l2L2Pressure = data[3]
	dt = data[5]
	
	print("HI")
	print(max_pressure_error)
	
	f.close()
	return [counter,l2L2,l2H1,l2L2Pressure,dt,max_pressure_error]

#Plotting Method

def compileErrors(fName,dtList):
	counter = np.array([])
	tol = np.array([])
	l2L2 = np.array([])
	l2H1 = np.array([])
	l2L2Pressure = np.array([])
	dts = np.array([])
	rejections = np.array([])
	max_pressure_errors=np.array([])
	for dt in dtList:
		[countertemp,l2L2temp,l2H1temp,l2L2Pressuretemp,currentDt,max_pressure_error] = getWorkData(fName + str(dt)+'.txt')
		counter = np.append(counter,countertemp)
		l2L2 = np.append(l2L2,l2L2temp)
		l2H1 = np.append(l2H1,l2H1temp)
		l2L2Pressure = np.append(l2L2Pressure,l2L2Pressuretemp)
		dts = np.append(dts,currentDt)
		max_pressure_errors = np.append(max_pressure_errors,max_pressure_error)
	return [counter,l2L2,l2H1,l2L2Pressure,dts,max_pressure_errors]

def writeErrorsToFile(fName,dtList,outputfname):
	[counter,l2L2,l2H1,l2L2Pressure,dts,max_pressure_error] = compileErrors(fName,dtList)
	f = open(outputfname,'w')
	f.write('Junk\n')
	f.write('dt, l2L2 error, l2H1 error, l2L2 Pressure error, Max Pressure Error \n')
	for i, dt in enumerate(dts):
		output = "\n" + str(dt) + "," + str(l2L2[i]) + "," +str(l2H1[i])  + "," + str(l2L2Pressure[i]) + "," + str(max_pressure_error[i])
		f.write(output)
		
	f.close()
		
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


dtList = [0.8,0.4,0.2,0.1,0.05, 0.025]
dtList=[0.2,0.1, 0.05, 0.025, 0.0125, 0.00625]
#[counter,l2L2,l2H1,l2L2Pressure,dt] = compileErrors('order-1-dt-',dtList)
#order 1 is bdf2
writeErrorsToFile('convergenceTestOrder4/order-1-dt-',dtList,'convergenceTestOrder4/order-4.txt')
writeErrorsToFile('convergenceTestOrder2/order-1-dt-',dtList,'convergenceTestOrder2/order-2.txt')
writeErrorsToFile('convergenceTestOrder3/order-1-dt-',dtList,'convergenceTestOrder3/order-3.txt')
#writeErrorsToFile('pressure-order-2-dt-',dtList,'pressure-order-2.txt')
#writeErrorsToFile('pressure-order-3-dt-',dtList,'pressure-order-3.txt')
#writeErrorsToFile('pressure-order-4-dt-',dtList,'pressure-order-4.txt')

#print(dt)
"""
plt.loglog(work,l2L2,'r:',marker = 'o',label = 'BE')
[counter,tol,l2L2,l2H1,l2L2Pressure,rejections] = compileErrors('second.txt',-6,-1)
work = counter + rejections
plt.loglog(work,l2L2,'b--',marker = 'o', label ='VSVO2')
[counter,tol,l2L2,l2H1,l2L2Pressure,rejections] = compileErrors('third.txt',-6,-1)
work = counter + rejections
plt.loglog(work,l2L2,'g-.',marker = 's',label ='VSVO3')
[counter,tol,l2L2,l2H1,l2L2Pressure,rejections] = compileErrors('home.txt',-6,-1)
print([counter,tol,l2L2,l2H1,l2L2Pressure,rejections])
work = counter + rejections
plt.loglog(work,l2L2,marker = '^',label ='VSVO4')
plt.xlabel("Work")
plt.ylabel("Velocity Error")
plt.legend()
plt.show()


"""
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
