# CUDA Monte Carlo Pi Estimation

This project is a high-performance implementation of the Monte Carlo method to estimate the value of Pi using NVIDIA's CUDA platform.

## Description

The program simulates throwing darts at a 1x1 square board that contains a quarter circle with a radius of 1. The ratio of darts that land inside the circle to the total number of darts thrown is used to estimate the area of the quarter circle (π/4), which is then used to calculate Pi.

The simulation is parallelized across thousands of GPU threads to generate billions of random points, leading to a highly accurate estimation of Pi in a short amount of time.

## Prerequisites

To build and run this project, you will need:
1.  An **NVIDIA GPU** with CUDA support.
2.  The **NVIDIA CUDA Toolkit** (v12.x recommended).
3.  **Visual Studio 2022** (or 2019) with the "Desktop development with C++" workload installed.

## Results

The results were obtained from the following:
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU
- Threads per block: 256
- Number of blocks: 768
- Samples per thread: 5086264
- Total samples requested: 1000000000000 (1 trillion)
- Total samples actual:    1000000192512

Points in circle: 785398375775

Kernel execution time: 10982.243164062500 ms (10.98 seconds)

Estimated Pi = 3.141592898306

Actual Pi    = 3.141592653590

Error        = 0.000000244716
