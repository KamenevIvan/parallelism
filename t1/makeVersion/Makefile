CXX = g++
CXXFLAGS = -std=c++14 -O2

USE_DOUBLE ?= 0

ifeq ($(USE_DOUBLE), 1)
    CXXFLAGS += -DUSE_DOUBLE
endif

TARGET = SinSum

all: $(TARGET)

$(TARGET): sinSum.cpp
	$(CXX) $(CXXFLAGS) -o $(TARGET) sinSum.cpp

clean:
	rm -f $(TARGET)