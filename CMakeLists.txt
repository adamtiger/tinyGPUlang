cmake_minimum_required(VERSION 3.22)

project(TinyGPUlang)
message("Start generating!")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(CUDA 12.0 REQUIRED)
set(CUDA_LIBRARIES PUBLIC ${CUDA_LIBRARIES} )     # To avoid "target must be either all-keyword or all-plain error" 

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_subdirectory(tinyGPUlang )
add_subdirectory(test        )

message("Finished")
