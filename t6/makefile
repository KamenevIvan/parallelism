# Размеры по умолчанию
# NX ?= 128
# NY ?= 128
# EPS ?= 1e-6
# MAX_ITER ?= 1000000

# # Компилятор и флаги
# CXX = pgc++
# CXXFLAGS = -acc -Minfo=all -O2 -std=c++14
# LDFLAGS = -lboost_program_options

# # Исходники и цели
# TARGET = solver
# SRC = main.cpp

# all: $(TARGET)

# $(TARGET): $(SRC)
# 	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC) $(LDFLAGS)

# run: $(TARGET)
# 	./$(TARGET) --nx=$(NX) --ny=$(NY) --eps=$(EPS) --max_iter=$(MAX_ITER)

# clean:
# 	rm -f $(TARGET) result.dat

#make run NX=256 NY=256 EPS=1e-5 MAX_ITER=100000

# ACCEL ?= gpu
# CXX = nvc++
# CXXFLAGS = -std=c++17 -O3 -acc=$(ACCEL) -Minfo=accel

# all: solver2

# solver2: heatSolverParal.cpp
# 	$(CXX) $(CXXFLAGS) -o solver2 heatSolverParal.cpp -lboost_program_options

# clean:
# 	rm -f solver2 result.dat

cpu_sequential:
	pgc++ -o task6_cpu -lboost_program_options -acc=host -Minfo=all -I/opt/nvidia/hpc_sdk/Linux_x86_64/23.11/cuda/12.3/include task6_cpu.cpp
	./task6_cpu --size=256 --accuracy=0.000001 --max_iterations=1000000

cpu_multicore:
	pgc++ -o task6_multicore -lboost_program_options -acc=multicore -Minfo=all -I/opt/nvidia/hpc_sdk/Linux_x86_64/23.11/cuda/12.3/include task6_cpu.cpp
	./task6_multicore --size=256 --accuracy=0.000001 --max_iterations=1000000

gpu:
	pgc++ -o task6_gpu -lboost_program_options -acc=gpu -Minfo=all -I/opt/nvidia/hpc_sdk/Linux_x86_64/23.11/cuda/12.3/include task6_gpu.cpp
	./task6_gpu --size=128 --accuracy=0.000001 --max_iterations=1000000
profile:
	nsys profile --trace=nvtx,cuda,openacc --stats=true ./task6_gpu --size=512 --accuracy=0.0001 --max_iterations=100