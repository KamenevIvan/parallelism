CC = g++
CFLAGS = -Wall -Wextra -O2 -std=c++17 -fopenmp
TARGETS = program1 program2 program3

all: $(TARGETS)

program1: t2_3.cpp
	$(CC) $(CFLAGS) $^ -o $@

program2: t2_3_v1.cpp
	$(CC) $(CFLAGS) $^ -o $@

program3: t2_3_v2.cpp
	$(CC) $(CFLAGS) $^ -o $@

run1: program1
	./program1

run2: program2
	./program2

run3: program3
	./program3

clean:
	rm -f $(TARGETS)
