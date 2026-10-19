# MonteCUDA — CUDA Monte Carlo Pi Estimation

A high-performance implementation of the Monte Carlo method to estimate the value of Pi using NVIDIA's CUDA platform.

## Description

The program simulates throwing darts at a 1x1 square board that contains a quarter circle with a radius of 1. The ratio of darts that land inside the circle to the total number of darts thrown is used to estimate the area of the quarter circle (π/4), which is then used to calculate Pi.

The simulation is parallelized across thousands of GPU threads to generate billions of random points, leading to a highly accurate estimation of Pi in a short amount of time.

## Prerequisites

To build and run this project, you will need:
1.  An **NVIDIA GPU** with CUDA support (compute capability 7.5+).
2.  The **NVIDIA CUDA Toolkit** (v12.x recommended).
3.  **CMake** 3.18 or later.
4.  A C++ compiler — **Visual Studio 2022** (or 2019) with the "Desktop development with C++" workload on Windows, or **GCC/Clang** on Linux.

## Building

```bash
# Configure the project (creates a build/ directory)
cmake -B build

# Build the executable
cmake --build build --config Release
```

## Running

```bash
# Run with default sample count (~1 billion)
./build/montecuda

# Run with a custom sample count (e.g., 1 trillion)
./build/montecuda 1000000000000
```

On Windows with Visual Studio, the executable will be at `build\Release\montecuda.exe`:

```powershell
.\build\Release\montecuda.exe 1000000000000
```

## Results

The results below were obtained on:
- **GPU:** NVIDIA GeForce RTX 4060 Laptop GPU
- **Threads per block:** 256
- **Number of blocks:** 768
- **Samples per thread:** 5,086,264
- **Total samples requested:** 1,000,000,000,000 (1 trillion)
- **Total samples actual:** 1,000,000,192,512

| Metric | Value |
|---|---|
| Points in circle | 785,398,375,775 |
| Kernel execution time | 10,982 ms (~11 seconds) |
| Estimated Pi | 3.141592898306 |
| Actual Pi | 3.141592653590 |
| Error | 0.000000244716 |

## Project Structure

```
montecuda/
├── CMakeLists.txt          # Build configuration
├── README.md
├── .gitignore
└── src/
    └── pi_monte_carlo.cu   # CUDA kernel and main program
```
