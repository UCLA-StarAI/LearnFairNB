cmake_minimum_required(VERSION 3.0.0)
project(NB_Calculator VERSION 0.1.0)

set (CMAKE_CXX_STANDARD 11)

add_executable(NB_Calculator 
                                main.cpp 
                                naive_bayes.cpp 
                                pattern_finder.cpp)