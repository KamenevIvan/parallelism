cmake_minimum_required(VERSION 3.10)
project(SinSum)

set(CMAKE_CXX_STANDARD 14)

option(USE_DOUBLE "Use double precision" OFF)

if(USE_DOUBLE)
    add_definitions(-DUSE_DOUBLE)
endif()

add_executable(SinSum sinSum.cpp)