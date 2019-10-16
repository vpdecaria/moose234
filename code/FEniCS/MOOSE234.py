"""
This is a fully coupled, fully implicity Backward Euler Code to solve NSE in a periodic box with an exact solution.

An exact solution to the Taylor Green Vortex in 2D is given by
u = (F(t)cos(x)sin(y), -F(t)cos(y)sin(x))
p = -(1/4)F(t)^2 (cos(2x) + cos(2y))

No boundary conditions because periodic domain.
"""

from __future__ import print_function
from fenics import *

import numpy as np
#import matplotlib.pyplot as plt

from types import ModuleType

#from scipy import linalg
#from stiffOde import *
import bdf


import timefilters as tf

import time as timer_python

import argparse
from subprocess import check_output


comm = MPI.comm_world
mpiRank = MPI.rank(comm)

#------------------------------------ CONSOLE INPUT --------------------------------------------------

parser = argparse.ArgumentParser(description='An implementation of the Embedded VSVO Backward Euler method \
	tested on a periodic domain with an exact quasi periodic solution.')

parser.add_argument('-f','--filters', help ="Run the test using up to the filter number specified by \
	this argument. An input of 0 is unmodified Backward Euler. The default selection is\
	4, which allows for the calculation of the estimator of the fourth order method.", type = int,default = 4)
parser.add_argument('--pressure', help ="Enabling this option will also apply time filters to the pressure of the same order specified in -f.",action="store_true")
parser.add_argument('-o','--output', help ="The file name for the text file containing the errors with respect to delta t. Default file name is date-time.txt",type =str, default = "errors/temp/" + str(check_output(['date','+%Y-%m-%d_%R:%S'])).strip()+".txt")
parser.add_argument('-t','--tolerance', help ="The tolerance used to adapt the step size. Right now, it is just based on error committed per time step, but can be made more sophisticated later.",type =np.float64, default = 1e-3)
parser.add_argument('-r','--ratio', help ="The maximum step size ratio allowed, where step size ratio is (step size at time n divided by step size at time n minus 1) The default value is 2.",type =np.float64, default = 2)
parser.add_argument('--constant', help ="Enabling this will disable adaptivity altogether.",action="store_true")
parser.add_argument('-k','--startingStepSize', help ="The initial step size taken.",type =np.float64, default = 0.000001)
parser.add_argument('--forcefilter', help ="Forces the exclusive use of the filter passed to argument f.",action="store_true")
parser.add_argument('-p','--problem', help ="The name of a user created python module that contains \
all the information necessary to run a specific test problem, including mesh, boundary conditions,\
					body forces, exact solutions, etc. There is a certain syntax that I need\
					to specify eventually. ",type =str, default = 'quasi_periodic_problem')
parser.add_argument('--bdforder', help ="The order of the bdf_method", type = int,default = 3)

parser.add_argument('--paraview', help ="Output name for pvd and vtu files",type =str, default = "pvd/tmp")
parser.add_argument('--parfreq', help ="Frequency with respect to delta t to take paraview snapshots.", type = int,default = 1000000)

parser.add_argument('--error', help ="Evaluate Error norms.",action="store_true")

parser.add_argument('--vo', help ="Which orders to use, such as 2, 23, 3, 34, 234", type = int,default = 3)


parser.add_argument('--nopics', help ="Don't write paraview output.",action="store_true")

args = parser.parse_args()

calculate_errors = args.error

writePVD = args.nopics

orders_to_use = args.vo

pviewOut = args.paraview

writePVD = True


if(args.nopics):
	writePVD = False


paraview_frequency = args.parfreq

exec('from '+str(args.problem) +' import *')
#from problem import *

print("Using filter number "+ str(args.filters))
filtersToUse = args.filters
if(args.pressure):
	print("Also filtering pressue.")
	
filterPressure = args.pressure

constantStepSize = args.constant
if(constantStepSize):
	print("Constant step size of " +str(args.startingStepSize)+ ". Adaptivity is turned off.")
	
print("Printing to file " + args.output)

tolerance = args.tolerance

maxRatio = args.ratio

forcefilter = args.forcefilter



#Initialize quantities related to adaptivity
EstVector = np.ones(3)*1e54
tempSolutions = ["Intermediate","Filtered","Solution","Values"]
safetyFactor = 0.9
numOfFailures = 0
minStepSize = 1e-12

errorfName = args.output
errorfile = open(errorfName, 'w')
output = "T final =" + str(T) +", Filters Used "+ str(filtersToUse) + str(filterPressure) +'\n'
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

Ts = np.array([-j*dt for j in range(total_num_steps,-1,-1)])


# Define variational problem
#f = f(t)


#Initial Conditions

print(Ts)


# Define trial and test functions
dw    = TrialFunction(W)

Gateux = TrialFunction(W)

PDE_rhs = TrialFunction(W)

PDE_rhs_u , PDE_rhs_p  = split(PDE_rhs)

Gateux_u , Gateux_p  = split(Gateux)

Gateux_   = Function(W)
Gateux_u_,Gateux_p_ = split(Gateux_)

PDE_rhs_   = Function(W)
PDE_rhs_u_ , PDE_rhs_p_  = split(PDE_rhs_)


v,q  = TestFunctions(W)

# Define functions
w   = TrialFunction(W)  # current solution

w_ = Function(W)
(u_,p_) = split(w_)

w_n  = Function(W)  # solution from previous converged step
w_extrap  = Function(W)  # solution from previous converged step

# Split mixed functions
#du, dp = split(du)
u,  p  = split(w)

"""
#Create some functions to solve for the 4th order estimate
#in a decoupled manner

v_ = TestFunction(V_)
PDE_rhs_c_decoupled = TrialFunction(V_)
PDE_rhs_mu_decoupled = TrialFunction(V_)
PDE_rhs_c_decoupled_ = Function(V_)
PDE_rhs_mu_decoupled_ = Function(V_)
gateux_c_decoupled = TrialFunction(V_)
gateux_mu_decoupled = TrialFunction(V_)
gateux_c_decoupled_ = Function(V_)
gateux_mu_decoupled_ = Function(V_)
"""


#u_n = np.array([exact(Ts[j]) for j in range(total_num_times)])

##u.vector()[:] = random.random(u.vector().size())
#w.vector()[:] = np.load("u"+str(4)+".npy")
#w_n.vector()[:] = u.vector().get_local()
w_n = [Function(W) for j in range(total_num_times)]


#for i in range(total_num_times):
#	w_n[i].vector()[:] = np.load("u"+str(i)+".npy")

u_n = [split(w_n[i])[0] for i in range(total_num_times)]

p_n = [split(w_n[i])[1] for i in range(total_num_times)]

p_temp = Function(W.sub(1).collapse())

#Initialize
for i,ttt in enumerate(Ts):
	assign(w_n[i].sub(0),interpolate(get_u_exact(ttt),W.sub(0).collapse()))
	assign(w_n[i].sub(1),interpolate(get_p_exact(ttt),W.sub(1).collapse()))


#Initialize temporal errors
l2L2_error = 0
l2L2 = 0



#initialize temp solutions
y_2 = Function(W)
y_3 = Function(W)
y_4 = Function(W)
y_4_est = Function(W)
#split variables
uy_4,py_4  = split(y_4)


time_filter = Function(W)
interp_error = Function(W)

#Create  Left hand sides for recovering the weak rhs and weak directional
#derivative
#of rhs of original pde
A_Gateux = assemble(dot(Gateux_u,v)*dx + Gateux_p*q*dx)

#A_PDE_rhs = assemble()

tOld = 0
#TIME STEPPING
output = "\n Time, StepSize, FilterUsed, NormU, NormP ,NormUExact, NormPExact, Uerror, Perror"
errorfile.write(output)



#solution = np.array(u_n[total_num_steps])


times = np.array([0])
newton_tol = 1e-8
maxIter = 50


dt_fixed= dt

#Give J a default value of 1 in case other bdfs are used
J=1


#Compute measure of domain
measure = assemble(Constant(1.0)*dx(mesh))

# Define convection term

def convect(u,v,w):
	return dot(dot(u, nabla_grad(v)), w)*dx

# Define skew-symmeterized forn
def b(u,v,w):
	return convect(u,v,w)+0.5*div(u)*dot(v,w)*dx

for j in range(total_num_steps):
	Ts[j] = Ts[j+1]

#Initialize error quanities
exact_vel_norm_sq = 0
L2_error = 0

l2L2_error = 0
l2L2 = 0
l2L2_error_pressure = 0
l2L2_pressure = 0
l2H1 = 0
l2H1_error = 0



step_counter = 0

if writePVD:
	file = File(pviewOut + ".pvd")
	#for j, tt in enumerate(Ts):
	#	file << (u_n[j].sub(0),tt)

#solver = KrylovSolver('bicgstab', "jacobi")
solver = KrylovSolver('gmres', "jacobi")
solver.parameters["relative_tolerance"] = 1e-7

solverEst = KrylovSolver('cg', "jacobi")
solverEst.parameters["relative_tolerance"] = 1e-6
solverEst.set_operator(A_Gateux)

#print(info(LinearVariationalSolver.default_parameters(), True))
loop_timer = timer_python.time()

while (tOld < T-1e-15):
	if(mpiRank == 0):
		print("Current dt - ", dt)
	#dt = dt_fixed + 5./6*dt_fixed*np.sin(t)
	t = tOld + dt
	
	#for j in range(total_num_steps):
	#	Ts[j] = Ts[j+1]
	Ts[total_num_steps] = t
	
	# Update current time
	#u_exact = exact(t)
	f = get_f(t)
	bcs = get_bcs(t)
	
	
	
	#Get new bdf coefficients
	[alpha,differences,eta] = bdf.bdf_coefficients_and_differences(Ts,bdf_order)
	
	#Make alphas dolfin constants. This keeps the JIT from 
	#being called every time the step sizes change.
	alpha_c = [Constant(alpha[i]) for i in range(len(alpha))]
	
	
	#Form guess for newton
	extrap_order = 4
	w_extrap.vector()[:] = -sum(differences[extrap_order][i-1]*w_n[i].vector() for i in range(1,total_num_times))/differences[extrap_order][total_num_steps]
	w_.vector()[:] = w_n[total_num_steps].vector()

	F = alpha_c[total_num_steps]*(dot(u,v)*dx) \
		+ dot(sum(alpha_c[i-1]*u_n[i] for i in range(1,total_num_times)),v)*dx \
		 +  b(w_extrap.sub(0),u,v)+nu*inner(nabla_grad(u), nabla_grad(v))*dx -p*div(v)*dx\
		  + div(u)*q*dx \
		  -dot(f,v)*dx \

	A = assemble(lhs(F))
	bcs = get_bcs(0)

	b_rhs = assemble(rhs(F))
	for bc in bcs:
		bc.apply(A)
		bc.apply(b_rhs)
	
	#Configure Solver
	
	
	#solver = KrylovSolver('bicgstab', "jacobi")
	#solver.parameters["relative_tolerance"] = 1e-6
	"""
	solver.set_operator(A)
	num_krylov_iterations = solver.solve(w_.vector(),b_rhs)
	
	print("Krylov solver converged in ", num_krylov_iterations)
	"""
	
	solve(A,w_.vector(),b_rhs)
	
	#Make pressure mean zero
	p_temp =  w_.split(True)[1]
	p_temp.vector()[:] = p_temp.vector().get_local() - assemble(p_temp/measure*dx)*np.ones_like(p_temp.vector().get_local())
	assign(w_.sub(1), p_temp)
	
	

	print(EstVector)

	
	if (bdf_order == 3):
		#This code can be used wfor any bdf, but not necessarily with vsvo capability
		if(orders_to_use == 2 or orders_to_use == 23 or orders_to_use == 234):
			#Use the stabilizing second order filter
			filter_coefficients = 9./125*differences[3]/differences[3][total_num_steps]
			#time_filter = (filter_coefficients[total_num_steps]*y_3  + np.sum(filter_coefficients[i-1]*w_n[i] for i in range(1,total_num_times)) )
			time_filter.vector()[:] = (filter_coefficients[total_num_steps]*w_.vector()  \
				+ sum(filter_coefficients[i-1]*w_n[i].vector() for i in range(1,total_num_times)) )
			
			EstVector[0] = norm(time_filter.sub(0),'L2',mesh)
			y_2.vector()[:] = w_.vector() + time_filter.vector()
			
			#down_filter = differences[3]/differences[3][total_num_steps]
			#y_2 = y_3 + 9./125*( down_filter[total_num_steps]*y_3  + np.sum(down_filter[i-1]*u_n[i] for i in range(1,total_num_times)) )
			#EstVector[0] = linalg.norm(9./125*( down_filter[total_num_steps]*y_3  + np.sum(down_filter[i-1]*u_n[i] for i in range(1,total_num_times)) ))
		if(orders_to_use == 23 or orders_to_use == 3 or \
				orders_to_use == 34 or orders_to_use == 234 or orders_to_use == 4 or orders_to_use == 24):
			#Construct the filter to go up an order
			#higher_order_filter = eta*differences[4]
	
			#EstVector[1] = linalg.norm(higher_order_filter)
			#y_4 = y_3 - (higher_order_filter[total_num_steps]*y_3 \
			#	+ np.sum(higher_order_filter[i]*u_n[i+1] for i in range(total_num_times-1)))
				
			filter_coefficients = eta*differences[4]
			time_filter.vector()[:] = (filter_coefficients[total_num_steps]*w_.vector()  \
				+ sum(filter_coefficients[i-1]*w_n[i].vector() for i in range(1,total_num_times)) )

			if(orders_to_use != 4 and orders_to_use != 24):
				#EstVector[1] = norm(time_filter.sub(0),'L2',mesh)
				EstVector[1] = norm(time_filter.sub(0),'L2',mesh)
				print("HI")
			y_4.vector()[:] = w_.vector() - time_filter.vector()

			
			
			if(not constantStepSize and (orders_to_use == 4 or orders_to_use == 34 or orders_to_use == 234 or orders_to_use == 24)):
				#estimate LTE of 4th order method
				#need coefficients for bdf4
				[alpha,temp,temp] = bdf.bdf_coefficients_and_differences(Ts,bdf_order+1)
				#Need some way to estimate y really well. For now, do a completely ad hoc way of using
				#the solution for BDF3-4
				Extrap = y_4
				#The time filter replaced with the extrapolation of y
				#is part of the error estimation
				time_filter.vector()[:] = (filter_coefficients[total_num_steps]*Extrap.vector()  \
				+ np.sum(filter_coefficients[i-1]*w_n[i].vector() for i in range(1,total_num_times)) )
				
				
				"""
				
				solver_time = timer.clock()
				b_rhs = assemble(-nu*inner(nabla_grad(time_filter.sub(0)),nabla_grad(v))*dx\
					-b(uy_4,time_filter.sub(0),v) - b(time_filter.sub(0),uy_4,v)\
					)
				
				
				solve(A_Gateux,Gateux_.vector(),b_rhs, "cg")
				
				#plot(Gateux_.sub(0))
				#plt.show()
				
				
				#NOW solve for F
				b_rhs = assemble(-nu*inner(nabla_grad(uy_4),nabla_grad(v))*dx\
					-b(uy_4,uy_4,v)\
					+(p_)*div(v)*dx)
				solve(A_Gateux,PDE_rhs_.vector(),b_rhs, "cg")

				
				y_4_est.vector()[:] =  alpha[total_num_steps]*y_4.vector()  \
				+ np.sum(alpha[i-1]*w_n[i].vector() for i in range(1,total_num_times))  \
				-PDE_rhs_.vector() - Gateux_.vector()
				EstVector[2] = norm(y_4_est.sub(0),'L2',mesh)/alpha[total_num_steps] #This last division by alpha normalizes the error
				print(EstVector)
				print("This solver took ",timer.clock() - solver_time)
				#Make alphas dolfin constants. This keeps the JIT from 
				#being called every time the step sizes change
				"""
				alpha_c = [Constant(alpha[i]) for i in range(len(alpha))]
				
				#SIMPLER WAY TO DO IT
				"""
				b_rhs = assemble(alpha_c[total_num_steps]*dot(uy_4,v)*dx\
					+ dot(sum(alpha_c[i-1]*u_n[i] for i in range(1,total_num_times)),v)*dx \
					+nu*inner(nabla_grad(uy_4),nabla_grad(v))*dx\
					+b(uy_4,uy_4,v) - (p_)*div(v)*dx -dot(f,v)*dx\
					+nu*inner(nabla_grad(time_filter.sub(0)),nabla_grad(v))*dx\
					+b(uy_4,time_filter.sub(0),v) + b(time_filter.sub(0),uy_4,v))
				"""
				b_rhs = assemble(alpha_c[total_num_steps]*dot(uy_4,v)*dx\
					+ dot(sum(alpha_c[i-1]*u_n[i] for i in range(1,total_num_times)),v)*dx \
					+nu*inner(nabla_grad(uy_4),nabla_grad(v))*dx\
					+b(uy_4,uy_4,v) - (p_)*div(v)*dx -dot(f,v)*dx)
				
				solver_time = timer_python.clock()
				#solve(A_Gateux,PDE_rhs_.vector(),b_rhs, "cg")
				
				num_krylov_iterations = solverEst.solve(PDE_rhs_.vector(),b_rhs)
	
				print("Krylov solver converged in ", num_krylov_iterations)
				
				print("This solver took ",timer_python.clock() - solver_time)
				EstVector[2] = norm(PDE_rhs_.sub(0),'L2',mesh)/alpha[total_num_steps]
				
				#print(EstVector)
				#exit()
			
			

	#The line below is a hacky way to exclude solutions that don't satisfy the tolerance from consideration for picking the next step size.
	if not constantStepSize:
		TempEstVector = EstVector + ~(EstVector< tolerance)*1e20
		[knp1,J] = tf.pickSolutionMaxK(TempEstVector,tolerance,dt,safetyFactor,[2,3,4])
		knp1 = np.max([np.max([np.min([knp1,maxRatio*dt]),dt/2]),minStepSize])
		#Force them all to end at the same time
		knp1 = np.min([knp1,T-t])
	if forcefilter:
		J = filtersToUse

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
		
		print("Using order ",J+2)
		if(J==2):
			print(EstVector)
			#exit()
		
		if(J==0):
			#w_ = y_2
			assign(w_.sub(0), y_2.sub(0))
		elif(J==1):
			pass
		else:
			assign(w_.sub(0), y_4.sub(0))
			#w_ = y_4
			
		
		if(mpiRank == 0):
			print("Using order ",J+2)
			print('At Time t = %.6f' % (t))
		if(calculate_errors):
			u_exact = get_u_exact(t)
			u_exact_interpolated = interpolate(u_exact,W.sub(0).collapse())
			L2_error = errornorm(u_exact_interpolated, w_.sub(0),degree_rise = 0)
			
			p_exact = get_p_exact(t)
			p_exact_interpolated = interpolate(p_exact,W.sub(1).collapse())
			p_L2_error = errornorm(p_exact_interpolated, w_.sub(1),degree_rise = 0)

			#Update Temporal Error
			exact_vel_norm_sq =norm(u_exact,'L2',mesh)**2
			l2L2 += exact_vel_norm_sq*dt
			l2L2_error += L2_error**2*dt
			
			exact_pres_norm_sq = norm(p_exact,'L2',mesh)**2
			l2L2_pressure += exact_vel_norm_sq*dt
			l2L2_error_pressure += p_L2_error**2*dt
			
			if(mpiRank == 0):
				print('t = %.2f: L2_error = %.3g' % (t, L2_error))
				print('t = %.2f: L2_error_pressure = %.3g' % (t, p_L2_error))
			
		
		normU = norm(w_.sub(0),'L2',mesh)
		normP = norm(w_.sub(1),'L2',mesh)
		if(mpiRank == 0):
			print('t = %.2f: Norm U = %.3g' % (t, normU))
			print('t = %.2f: Norm P = %.3g' % (t, normP))
		

		output = "\n" + str(t) + "," + str(dt) + "," + str(J) + "," +str(normU)+ "," + \
		str(normP)+ ","+str(exact_vel_norm_sq**0.5)+ ","+ str(exact_pres_norm_sq**0.5) \
		+ "," +str(L2_error)+ ","+str(p_L2_error)
		errorfile.write(output)

		#plot(np.abs(w_.sub(1)-p_exact_interpolated))
		#plt.pause(.00000001)
		
		# Update previous solution
		for j in range(total_num_steps):
			w_n[j].vector()[:] = w_n[j+1].vector()
		w_n[total_num_steps].vector()[:] = w_.vector()
		
		
		

		#print(K)
		
		#Update times and solution
		
		#dt = dt*1.1
		
		tOld += dt
		
		for j in range(total_num_steps):
			Ts[j] = Ts[j+1]
		if not constantStepSize:
			dt = knp1
			k = knp1
			
		if(step_counter%paraview_frequency ==0 and writePVD):
			file << (w_.sub(0),t)

		step_counter+=1
	else:
		print("Failed FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF: Halving Time Step")
		#Update Estimates
		#saftey_factor_failed = 0.5
		saftey_factor_failed = 0.7
		[knp1,junk] = tf.pickSolutionMaxK(EstVector,tolerance,dt,saftey_factor_failed,[2,3,4])
		#dt = np.max([dt/2.,knp1])
		dt = knp1
		k = dt
		numOfFailures += 1
		#Force all simulations to end at the same time
		knp1 = np.min([knp1,T-t])
		#exit()
	
elapsed_time = timer_python.time()-loop_timer
print("Main loop took ",elapsed_time ," seconds.")

#Calculate final errors

errorfile.write("\nEND\ntolerance, l2L2 error, l2H1 error, l2L2 Pressure error, Number Of Rejections, Starting Step Size, Elapsed Time")
relative_l2L2_error = np.sqrt(l2L2_error)/np.sqrt(l2L2)
relative_l2L2_error_pressure = np.sqrt(l2L2_error_pressure)/np.sqrt(l2L2_pressure)
relative_l2H1_error = np.sqrt(l2H1_error)/np.sqrt(l2H1)
print("l2L2 error:",relative_l2L2_error)
print("l2L2P error:",relative_l2L2_error_pressure)

#Calculate final errors
output = "\n" + str(tolerance) + "," + str(relative_l2L2_error) + "," +str(relative_l2H1_error)  + "," + str(relative_l2L2_error_pressure)+ "," + str(numOfFailures) + "," + str(args.startingStepSize)+ "," + str(elapsed_time)
errorfile.write(output)
errorfile.close()

