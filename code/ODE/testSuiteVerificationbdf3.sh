
ode='OscillatoryOde'

python_script='VSVOODEBDF3.py'
#python_script='VSVOBDF3_old.py'

let order=2
for vo in  2 3 4
	do
	for dt in '0.2' '0.1' '0.05' '0.025' '0.0125' '0.00625'
	do
		python ${python_script} -p ${ode} -k $dt -f 0 --constant --forcefilter -o errors/convergenceVerificationBDF3/order-$order-dt-$dt.txt --bdforder 3 --vo $vo

	done
	let "order = order +1"
	

done


#Calling BDF2 "order 1"
let order=1

for dt in '0.2' '0.1' '0.05' '0.025' '0.0125' '0.00625'
do
	python ${python_script} -p ${ode} -k $dt -f 0 --constant --forcefilter -o errors/convergenceVerificationBDF3/order-$order-dt-$dt.txt --bdforder 2 

done

#Calling BDF4 "order 5"
let order=5

for dt in '0.2' '0.1' '0.05' '0.025' '0.0125' '0.00625'
do
	python ${python_script} -p ${ode} -k $dt -f 0 --constant --forcefilter -o errors/convergenceVerificationBDF3/order-$order-dt-$dt.txt --bdforder 4 

done
