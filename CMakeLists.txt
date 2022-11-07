#cmake_minimum_required(VERSION 3.22)

#project(CIVL LANGUAGES CUDA)

#set(CMAKE_CUDA_STANDARD 14)

#add_library(${PROJECT_NAME}
#        library.cu)

#set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Recommended build using Visual Studio toolset.
cmake_minimum_required(VERSION 3.0)
project(CIVL LANGUAGES CUDA CXX)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)

add_subdirectory("Lib")
add_subdirectory("Examples")