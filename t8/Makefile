TARGET = task8_v2

all: $(TARGET)

$(TARGET): task8_v2.cu
	nvcc task8_v2.cu -o $(TARGET) -O3 -std=c++17 -lboost_program_options -lcudart

profile: $(TARGET)
	nsys profile --trace=nvtx,cuda --stats=true ./$(TARGET) --max_iterations=50
