
let mpi_threads=1

for vo in 3
do
	for dt in  '0.2' '0.1' '0.05' '0.025' '0.0125' '0.00625'
	do
		let order=filters+1
		mpirun -np $mpi_threads python3 MOOSE234.py --constant --forcefilter -p taylor_green_problem -k $dt --error -o errors/convergenceTestOrder$vo/order-$order-dt-$dt.txt --vo $vo
	done

done
