CC = g++
CFLAGS = -fopenmp -Wall -Wextra -O2

TARGET = program
SRC = t2_1.cpp

all: $(TARGET)

$(TARGET): $(SRC)
	$(CC) $(CFLAGS) -o $(TARGET) $(SRC)

clean:
	rm -f $(TARGET)