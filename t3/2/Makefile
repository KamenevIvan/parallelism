CXX = g++


CXXFLAGS = -Wall -Wextra -O2 -std=c++17 -pthread


TARGET = program2


SRC = t3_2.cpp


all: $(TARGET)


$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(SRC)


clean:
	rm -f $(TARGET)


run: $(TARGET)
	./$(TARGET)


rebuild: clean all