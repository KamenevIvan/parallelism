# Размеры по умолчанию
NX ?= 128
NY ?= 128
EPS ?= 1e-6
MAX_ITER ?= 1000000

# Компилятор и флаги
CXX = pgc++
CXXFLAGS = -acc -Minfo=all -O2 -std=c++14
LDFLAGS = -lboost_program_options

# Исходники и цели
TARGET = solver
SRC = main.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC) $(LDFLAGS)

run: $(TARGET)
	./$(TARGET) --nx=$(NX) --ny=$(NY) --eps=$(EPS) --max_iter=$(MAX_ITER)

clean:
	rm -f $(TARGET) result.dat

#make run NX=256 NY=256 EPS=1e-5 MAX_ITER=100000