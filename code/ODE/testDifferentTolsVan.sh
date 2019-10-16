
ode='vanDerPolODE'

python_script='VSVOODEBDF3.py'
#python_script='VSVOBDF3_old.py'

let order=2
for vo in   234 34 4 # 24 # 2 3 23
	do
	for tol_exponent in  {4..11}
#	for tol_exponent in  {12..18}
		do
		python ${python_script} -p ${ode} -k 0.00000001 -o errors/${ode}/1e-${tol_exponent}-order-${vo}.txt --bdforder 3 --vo $vo -t 1e-$tol_exponent
		#python ${python_script} -p ${ode} -k 0.00000001  --bdforder 3 --vo $vo -t 1e-$tol_exponent

	done

done
