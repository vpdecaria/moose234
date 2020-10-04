
ode='protheroOde'

python_script='VSVOODEBDF3.py'
#python_script='VSVOBDF3_old.py'

let order=4
for vo in   2 3 4
	do
	for dt in '0.2' '0.1' '0.05' '0.025' '0.0125' '0.00625'
	do
		python ${python_script} -p ${ode} -k $dt -f 0 --constant --forcefilter -o errors/protheroOde/order-$vo-dt-$dt.txt --bdforder 3 --vo $vo

	done
	let "order = order +1"
	

done

