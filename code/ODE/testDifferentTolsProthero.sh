
ode='protheroOde'

mkdir errors/${ode}

python_script='VSVOODEBDF3.py'
#python_script='VSVOBDF3_old.py'

let order=2
for vo in   3 234 2 4 #24 # 2 3 23
	do
	for tol_exponent in  {3..14}
		do
		python ${python_script} -p ${ode} -k 0.0000000001 -o errors/${ode}/1e-${tol_exponent}-order-${vo}.txt --bdforder 3 --vo $vo -t 1e-$tol_exponent

	done

done
