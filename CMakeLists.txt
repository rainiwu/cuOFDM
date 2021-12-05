cmake_minimum_required(VERSION 3.12) 
project(cuOFDM LANGUAGES CXX CUDA)              
set(CMAKE_CXX_STANDARD 17)           
set(CMAKE_CUDA_STANDARD 11)


include_directories(hdr)
add_subdirectory(src)
