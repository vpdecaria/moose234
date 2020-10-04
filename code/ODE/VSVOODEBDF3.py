"""
A modular, ODE version of MOOSE234 for solving first order systems of ODEs.
"""

from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

from types import ModuleType

from scipy import linalg
#from stiffOde import *
import bdf
import newton

import timefilters as tf

import argparse
from subprocess import check_output
#------------------------------------ CONSOLE INPUT ------------------------------------------------

parser = argparse.ArgumentParser(description='An implementation of the Embedded VSVO Backward Euler\
   method tested on a periodic domain with an exact quasi periodic solution.')

parser.add_argument('-f','--filters', help ="Run the test using up to the filter number specified\
    by this argument. An input of 0 is unmodified Backward Euler. The default selection is\
	4, which allows for the calculation of the estimator of the fourth order method.",\
    type = int,default = 4)
parser.add_argument('-o','--output', help ="The file name for the text file containing the errors \
	                with respect to delta t. Default file name is date-time.txt",\
	                type =str, default = "errors/temp/" + \
	                str(check_output(['date','+%Y-%m-%d_%R:%S'])).strip()+".txt")
parser.add_argument('-t','--tolerance', help ="The tolerance used to adapt the step size. Right now\
	                , it is just based on error committed per time step, but can be made more \
	                sophisticated later.",\
	                type =np.float64, default = 1e-3)
parser.add_argument('-r','--ratio', help ="The maximum step size ratio allowed, where step size\
                    ratio is (step size at time n divided by step size at time n minus 1) \
                    The default value is 2.",type =np.float64, default = 2)
parser.add_argument('--constant', help ="Enabling this will disable adaptivity altogether.",\
	                action="store_true")
parser.add_argument('-k','--startingStepSize', help ="The initial step size taken.",\
	                type =np.float64, default = 0.000001)
parser.add_argument('-m','--min', help ="Minimum stepsize alloweed",\
	                type =np.float64, default = 1e-15)
parser.add_argument('--forcefilter',\
                    help ="Forces the exclusive use of the filter passed to argument f.",\
                    action="store_true")
parser.add_argument('-p','--problem', help ="The name of a user created python module that contains\
                    all the information necessary to run a specific test problem, including mesh, \
                    boundary conditions, body forces, exact solutions, etc. There is a certain \
                    syntax that I need\
					to specify eventually. ",type =str, default = 'butcherOde')
parser.add_argument('--bdforder', help ="The order of the bdf_method", type = int,default = 3)

parser.add_argument('--vo', help ="Which orders to use, such as 2, 23, 3, 34, 234",\
                    type = int,default = 3)
parser.add_argument('--plot', help ="Plot the solution at the end.",\
	                action="store_true")
parser.add_argument('-s','--solution', help ="Solution file name.",\
	                type =str, default = "none" )

#----------------------------------- PARSING THE CONSOLE INPUT -------------------------------------
args = parser.parse_args()

orders_to_use = args.vo

exec('from '+str(args.problem) +' import *')
#from problem import *

print("Using filter number "+ str(args.filters))
filtersToUse = args.filters
	
constantStepSize = args.constant
if(constantStepSize):
	print("Constant step size of " +str(args.startingStepSize)+ ". Adaptivity is turned off.")
	
print("Printing to file " + args.output)

tolerance = args.tolerance

if(args.solution == "none"):
	save_solution = False
else:
	save_solution = True
	solution_filename = args.solution

maxRatio = args.ratio

forcefilter = args.forcefilter

#If exact solution comes from data. Move this function to it's own module later.

def getData(filename):

	f = open(filename,'r')

	#First 3 lines are header information

	data = np.array([])

	for j,line in enumerate(f):

		if line == 'END\n':
			break
		temp = line.strip().split(',')
		temp = [float(i) for i in temp]
		length = len(temp)
		data = np.append(data,temp)

	f.close()
	data = data.reshape((len(data)/length,length))
	return data


if(numerical_data):
	t_data = getData(t_data_file)
	y_data = getData(y_data_file)




#Initialize quantities related to adaptivity
EstVector = np.ones(3)*1e14
tempSolutions = ["Intermediate","Filtered","Solution","Values"]
safetyFactor = 0.9
numOfFailures = 0
minStepSize = args.min

errorfName = args.output
errorfile = open(errorfName, 'w')
output = "T final =" + str(T) +", Filters Used "+ str(filtersToUse) +'\n'
errorfile.write(output)

#Start dt loop

t = 0

dt = np.float64(args.startingStepSize)
k   = dt
K = np.ones(4)*dt

#vector of times. Earlier times at start of array

bdf_order = args.bdforder
bdf_num_times = bdf_order+1

total_num_steps=bdf_order + 1
total_num_times = total_num_steps+1

Ts = np.array([-j*dt for j in xrange(total_num_steps,-1,-1)])

if(numerical_data):
	Ts = np.array([t_data[j][0] for j in xrange(total_num_times)])
	t=Ts[total_num_steps]
	dt = Ts[total_num_steps] - Ts[total_num_steps-1]

#Initial Conditions

print(Ts)

u_n = np.array([exact(Ts[j]) for j in xrange(total_num_times)])


if(numerical_data):
	u_n = np.array([np.array([y_data[j]]).transpose() for j in xrange(total_num_times)])


#####  INITIALIZE ERROR QUANTITIES ####
l2L2_error = 0                        #
l2L2       = 0                        #                   
#######################################

#initialize temp solutions
y_2 = exact(0)
y_3 = exact(0)
y_4 = exact(0)

tOld = 0
#TIME STEPPING
output = "\n Time, StepSize, FilterUsed, NormU,NormUExact, Uerror"
errorfile.write(output)

solution = np.array(u_n[total_num_steps])


times = np.array([0])
newton_tol = 1e-9
maxIter = 50


dt_fixed= dt

#Give J a default value of 1 in case other bdfs are used
J=1

for j in range(total_num_steps):
	Ts[j] = Ts[j+1]

###########################################################
#                                                         #
#        LL         OOOOO      OOOOO     PPPPP            #
#        LL        O     O    O     O    P    P           #
#        LL        O     O    O     O    P    P           #
#        LL        O     O    O     O    PPPPP            #
#        LL        O     O    O     O    P                #
#        LL        O     O    O     O    P                #
#        LLLLLLLLL  OOOOO      OOOOO     P                #
#                                                         #
###########################################################

while (tOld < T):
	#dt = dt_fixed + 5./6*dt_fixed*np.sin(t)
	t = tOld + dt
	
	#for j in range(total_num_steps):
	#	Ts[j] = Ts[j+1]
	Ts[total_num_steps] = t
	
	# Update current time
	u_exact = exact(t)
	
	#Get new bdf coefficients
	[alpha,differences,eta] = bdf.bdf_coefficients_and_differences(Ts,bdf_order)
	
	F        =  lambda u :  alpha[total_num_steps]*u \
	         +  np.sum(alpha[i-1]*u_n[i] for i in xrange(1,total_num_times))   - f(t,u)
	J_newton =  lambda u :  alpha[total_num_steps]*np.identity(u_n[0].size)    - Jf(t,u)
	

	maxIter = 50
	y_3 = newton.newton(u_n[total_num_steps],F,J_newton,newton_tol,maxIter)
	
	if (bdf_order == 3):
		#This code can be used wfor any bdf, but not necessarily with vsvo capability
		if(orders_to_use == 2 or orders_to_use == 23 or orders_to_use == 234):
			#Use the stabilizing second order filter
			filter_coefficients = 9./125*differences[3]/differences[3][total_num_steps]
			#filter_coefficients = 9./125*np.array([0,-1,3,-3,1])
			"""
			filter_coefficients = 11./2*9./125*(Ts[total_num_steps] - Ts[total_num_steps-1])\
			*(Ts[total_num_steps] - Ts[total_num_steps-2])\
				*differences[3]/differences[3][total_num_steps]
			"""
			
			time_filter =(filter_coefficients[total_num_steps]*y_3  \
				        +np.sum(filter_coefficients[i-1]*u_n[i] for i in xrange(1,total_num_times)))
			EstVector[0] = linalg.norm(time_filter)
			y_2 = y_3 + time_filter
			
		if(        orders_to_use == 23 or orders_to_use == 3 \
				or orders_to_use == 34 or orders_to_use == 234\
			    or orders_to_use == 4  or orders_to_use == 24):
			#Construct the filter to go up an order
			#higher_order_filter = eta*differences[4]
				
			filter_coefficients = eta*differences[4]
			time_filter = (filter_coefficients[total_num_steps]*y_3  \
				+ np.sum(filter_coefficients[i-1]*u_n[i] for i in xrange(1,total_num_times)) )
			if(orders_to_use != 4 and orders_to_use != 24):
				EstVector[1] = linalg.norm(time_filter)
			y_4 = y_3 - time_filter
			
			if(not constantStepSize and (orders_to_use == 4   or orders_to_use == 34\
			                        or   orders_to_use == 234 or orders_to_use == 24)):
				#estimate LTE of 4th order method
				#need coefficients for bdf4
				[alpha,temp,temp] = bdf.bdf_coefficients_and_differences(Ts,bdf_order+1)
				#Need some way to estimate y really well. For now, do a completely ad hoc way of using
				#the solution for BDF3-4
				Extrap = y_4
				#The time filter replaced with the extrapolation of y
				#is part of the error estimation

				Est_3 = y_4-y_3
					
				EstVector[2] = linalg.norm((alpha[total_num_steps]*Extrap +\
				    np.sum(alpha[i-1]*u_n[i] for i in xrange(1,total_num_times))\
				    - f(t,Extrap))/alpha[total_num_steps])

	#The line below is a hacky way to exclude solutions that don't satisfy the tolerance from 
	#consideration for picking the next step size.
	if not constantStepSize:
		TempEstVector = EstVector + ~(EstVector< tolerance)*1e20
		[knp1,J] = tf.pickSolutionMaxK(TempEstVector,tolerance,dt,safetyFactor,[2,3,4])
		knp1 = np.max([np.max([np.min([knp1,maxRatio*dt]),dt/2]),minStepSize])
		knp1 = np.min([knp1,T-t])
		#exit()
	if forcefilter:
		J = filtersToUse
	print(J)
	if(constantStepSize or EstVector[J] < tolerance or np.abs(knp1 - minStepSize) < 1e-10):
		#u_=UTEMP[J]
		if(orders_to_use == 2):
			#u_ = y_2
			J=0
		elif(orders_to_use == 3):
			#u_ = y_3
			J=1
		elif(orders_to_use == 4):
			#u_ = y_4
			J=2
			
		if(J==0):
			u_ = y_2
		elif(J==1):
			u_ = y_3
		else:
			u_ = y_4
		print("dt "+str(dt))
		#Calculate Error
		L2_error = linalg.norm(u_ - u_exact)

		#Update Temporal Error
		exact_vel_norm_sq =linalg.norm(u_exact)**2
		l2L2 += exact_vel_norm_sq*dt
		l2L2_error += L2_error**2*dt

		print('At Time t = %.6f' % (t))


		print('t = %.2f: L2_error = %.3g' % (t, L2_error))
		
		normU = linalg.norm(u_)
		
		output = "\n" + str(t) + "," + str(dt) + "," + str(J) + "," +str(normU)+ "," + \
		str(0)+ ","+str(exact_vel_norm_sq**0.5)+ ","+ str(0) \
		+ "," +str(L2_error)+ ","+str(0)
		errorfile.write(output)

		#store solution
		#solution = np.append(solution,u_)
		solution = np.hstack((solution,u_))
		times = np.append(times,t)

		
		# Update previous solution
		for j in range(total_num_steps):
			u_n[j] = u_n[j+1]
		u_n[total_num_steps] = u_
		
		#Update times and solution
		
		tOld += dt
		
		for j in range(total_num_steps):
			Ts[j] = Ts[j+1]
		if not constantStepSize:
			dt = knp1
			k = knp1
		
	else:
		print("Failed FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF: Halving Time Step")
		#Update Estimates
		[knp1,junk] = tf.pickSolutionMaxK(EstVector,tolerance,dt,0.7,[2,3,4])
		dt = np.max([dt/2.,knp1])
		k = dt
		numOfFailures += 1
		#exit()

##################################################################################
#                                                                                #
#    EEEEEEEE  N     N   DDDDD          L         OOOOO      OOOOO     PPPPP     #                  
#    E         NN    N   D    D         L        O     O    O     O    P    P    #  
#    E         N N   N   D     D        L        O     O    O     O    P    P    # 
#    EEEEEEEE  N  N  N   D     D        L        O     O    O     O    PPPPP     #           
#    E         N   N N   D     D        L        O     O    O     O    P         #
#    E         N    NN   D    D         L        O     O    O     O    P         # 
#    EEEEEEEE  N     N   DDDDD          LLLLLLLL  OOOOO      OOOOO     P         #  
#                                                                                #                     
################################################################################## 	

final_error=L2_error
if(numerical_data):
	print('Final error at t = ')

	print(t_data[len(t_data)-1])
	exact_final = np.array([y_data[len(t_data)-1]]).transpose()
	final_error = linalg.norm(u_- exact_final)/linalg.norm(exact_final)
	print(final_error)



#Calculate final errors
errorfile.write("\nEND\ntolerance, l2L2 error, l2H1 error, l2L2 Pressure error, \
	Number Of Rejections, Starting Step Size,final error")
relative_l2L2_error = np.sqrt(l2L2_error)/np.sqrt(l2L2)
relative_l2L2_error_pressure = 11111
relative_l2H1_error = 1111
print("l2L2 error:",relative_l2L2_error)

#Calculate final errors
output = "\n" + str(tolerance)       + "," + str(relative_l2L2_error)          + "," \
       + str(relative_l2H1_error)    + "," + str(relative_l2L2_error_pressure) + "," \
       + str(numOfFailures)          + "," + str(args.startingStepSize)        + "," \
       + str(final_error)
errorfile.write(output)
errorfile.close()

#
if args.plot:
#	plt.plot(times,solution[0],marker = "x")
	if(len(exact(0)) == 1):
		plt.plot(times,solution,'k')
	else:
		#just first componenet
		plt.plot(times,solution[0],'k')
#	plt.plot(t_data,y_data[:,0])
#	plt.semilogy(times,solution[0])
#	plt.semilogy(t_data,y_data[:,0])
#	plt.plot(times,solution,marker = "x")
#	plt.plot(times,np.ones_like(solution))
	plt.show()
	plt.plot(times)
	plt.show()
	#print('Final Solution = ' + str(solution[0][len(solution[0])-1]))
#print('Final Solution = ' + str(solution[0][len(solution[0])-1]))
#print('Final Solution = ' + str(solution[1][len(solution[0])-1]))

print(len(times))
print(len(times)+numOfFailures)

if(save_solution):
	np.savetxt("y-" + solution_filename, solution)
	np.savetxt("t-" + solution_filename, t_data)
