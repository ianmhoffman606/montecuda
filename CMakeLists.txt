cmake_minimum_required(VERSION 3.18)
project(montecuda LANGUAGES CXX CUDA)

# Require C++17 for both host and device code
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Target CUDA architectures: Turing (75), Ampere (80, 86), Ada Lovelace (89)
set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89)

# Define the GPU (CUDA) executable
add_executable(montecuda src/pi_monte_carlo.cu)

# Link the curand library for random number generation
find_package(CUDAToolkit REQUIRED)
target_link_libraries(montecuda PRIVATE CUDA::curand)

# Define the CPU-only executable (no CUDA, no parallelization)
add_executable(montecuda_cpu src/pi_monte_carlo_cpu.cpp)

# Enable optimizations for Release builds (default)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
