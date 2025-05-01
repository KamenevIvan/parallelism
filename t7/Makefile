gpu_cublas:
	pgc++ -o task7 -lboost_program_options -acc=gpu -fast \
		-I/opt/nvidia/hpc_sdk/Linux_x86_64/23.11/cuda/12.3/include \
		-I/usr/local/cuda/include \
		-L/usr/local/cuda/lib64 \
		-I/opt/nvidia/hpc_sdk/Linux_x86_64/23.11/math_libs/12.3/targets/x86_64-linux/include \
		-L/opt/nvidia/hpc_sdk/Linux_x86_64/23.11/math_libs/12.3/targets/x86_64-linux/lib \
		-L/opt/nvidia/hpc_sdk/Linux_x86_64/23.11/cuda/12.3/lib64 \
		-lcublas -lcudart task7.cpp

clear:
	rm task7 *.nsys-rep *.sqlite

profile:
	nsys profile --trace=nvtx,cuda,openacc --stats=true ./task7 --max-iters=50


#./task7 --size 128 --accuracy 0.000001 --max-iters 1000000