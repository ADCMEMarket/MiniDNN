cmake_minimum_required(VERSION 3.5)
project(TF_CUSTOM_OP)
set (CMAKE_CXX_STANDARD 11)

include_directories(../include)
include_directories(../../../.julia/conda/3/Lib/Libraries/eigen3)

add_executable(main main.cpp)