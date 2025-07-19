# CUDA Monte Carlo Pi Estimation

This project is a high-performance implementation of the Monte Carlo method to estimate the value of Pi using NVIDIA's CUDA platform. It is designed to be built and run directly from Visual Studio Code on Windows.

## Description

The program simulates throwing darts at a 1x1 square board that contains a quarter circle with a radius of 1. The ratio of darts that land inside the circle to the total number of darts thrown is used to estimate the area of the quarter circle (π/4), which is then used to calculate Pi.

The simulation is parallelized across thousands of GPU threads to generate billions of random points, leading to a highly accurate estimation of Pi in a short amount of time.

## Features

- **Massively Parallel**: Leverages thousands of GPU threads for simulation.
- **High Performance**: Uses `cuRAND` for fast, high-quality parallel random number generation.
- **Efficient Reduction**: Implements an efficient parallel reduction using shared memory to sum up results within a thread block.
- **Performance-Aware**: Automatically scales the workload based on the number of multiprocessors on the GPU.
- **Accurate Timing**: Includes precise kernel execution timing using CUDA Events.
- **VS Code Integration**: Comes with pre-configured `tasks.json` and `launch.json` for one-press (`F5`) compilation and execution.

## Prerequisites

To build and run this project, you will need:
1.  An **NVIDIA GPU** with CUDA support.
2.  The **NVIDIA CUDA Toolkit** (v12.x recommended).
3.  **Visual Studio 2022** (or 2019) with the "Desktop development with C++" workload installed.
4.  **Visual Studio Code**.
5.  The **C/C++ Extension Pack** and **NVIDIA Nsight** extensions for VS Code.

## Building and Running

The project is configured to be built and run directly from Visual Studio Code.

1.  Open the project folder in VS Code.
2.  Open the `pi_monte_carlo.cu` file.
3.  Press **`F5`**.

This will automatically:
-   Compile the CUDA code using the build task defined in `.vscode/tasks.json`.
-   Run the resulting executable with the arguments specified in `.vscode/launch.json`.
-   Display the output in the VS Code terminal.

### Adjusting the Simulation

You can change the total number of samples for the simulation by editing the `args` array in `.vscode/launch.json`.

## Configuration Notes

The `.vscode` directory contains configuration files tailored for a specific Windows environment. If you encounter build errors (e.g., `cl.exe` not found), you may need to adjust the following paths to match your local installation:

-   **`.vscode/tasks.json`**: The `-ccbin` argument points to the `cl.exe` host compiler.
-   **`.vscode/c_cpp_properties.json`**: The `includePath` and `compilerPath` should match your CUDA Toolkit and Visual Studio versions.