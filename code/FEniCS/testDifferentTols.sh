
ode='taylor_green_problem'

python_script='MOOSE234.py'
#python_script='VSVOBDF3_old.py'

let order=2
for vo in  2 3 4 234 34 23 #24 3 23
	do
	for tol_exponent in  {1..8} #7
		do
		mpirun -np 4 python3 ${python_script} -p ${ode} -k 0.001 -o errors/${ode}/1e-${tol_exponent}-order-${vo}.txt --bdforder 3 --vo $vo -t 1e-$tol_exponent --error

	done

done
