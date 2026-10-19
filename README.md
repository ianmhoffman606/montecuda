# MonteCUDA — CUDA Monte Carlo Pi Estimation

A high-performance CUDA implementation of the Monte Carlo method for estimating $\pi$, benchmarked with a direct side-by-side comparison against single-threaded CPU execution.

> **Performance Highlight:** **1,068x GPU Speedup!**  
> Running **1 trillion random samples** completes in **~4.85 seconds** on an NVIDIA RTX 4060 Laptop GPU, compared to an estimated **~1.44 hours (5,184 seconds)** on an Intel Core i7-14700HX CPU.

---

## Performance Comparison (1 Trillion Samples)

Below are benchmark results obtained directly from running `montecuda.exe`:

- **GPU:** NVIDIA GeForce RTX 4060 Laptop GPU (Ada Lovelace, 24 SMs, 768 blocks, 256 threads/block = 196,608 concurrent threads)
- **CPU:** Intel(R) Core(TM) i7-14700HX (Single-threaded C++ baseline)
- **Requested Samples:** 1,000,000,000,000 (1 Trillion)
- **Actual Samples Processed:** 1,000,000,192,512

### Benchmark Table

| Metric | GPU (CUDA Parallelized) | CPU (Single-threaded) | Difference / Speedup |
|:---|:---|:---|:---|
| **Samples Processed** | 1,000,000,192,512 | 1,000,000,192,512 *(extrapolated from 100M)* | Same workload |
| **Execution Time** | **4,852.97 ms (~4.85 s)** | **5,184,578.00 ms (~1.44 hours)** | **1,068.33x faster** |
| **Throughput** | **~206.1 Billion samples/s** | **~192.9 Million samples/s** | **+205.9 Billion/s** |
| **Points in Circle** | 785,398,858,650 | 785,410,861,200 | Ratio $\approx \pi / 4$ |
| **Estimated $\pi$** | **3.141594829805** | **3.141642840000** | Accurate to 5-6 digits |
| **Actual $\pi$** | 3.141592653590 | 3.141592653590 | — |

```
============================================================
  COMPARISON: GPU vs CPU
============================================================
Metric                          GPU (CUDA)      CPU (Single)
  --------------------------------------------------------
Samples                      1000000192512     1000000192512
Time (ms)                          4852.97        5184578.00
Estimated Pi                  3.1415948298      3.1416428400
Points in circle              785398858650      785410861200

  >>> GPU speedup: 1068.33x faster than CPU <<<
```

---

## How It Works

1. **Monte Carlo Method:** Random points $(x, y)$ are sampled uniformly within a $1 \times 1$ unit square ($x, y \in [0, 1]$). Points satisfying $x^2 + y^2 \le 1$ lie inside a quarter circle of radius 1.
2. **Estimation Formula:** The area of the quarter circle is $\pi / 4$. Therefore:
   $$\pi \approx 4 \times \frac{\text{Points in Circle}}{\text{Total Samples}}$$
3. **GPU Parallelization Strategy:**
   - **cuRAND:** Each GPU thread maintains its own independent PRNG state (`curandState_t`) initialized with unique per-thread seeds.
   - **Block Shared Memory Reduction:** Within each thread block, counts are accumulated in shared memory using a tree reduction step, avoiding global memory atomic contention.
   - **Grid Dimension Tuning:** Grid size dynamically tunes to `SM_count * 32` blocks with 256 threads per block to maximize warp occupancy on NVIDIA architectures.

---

## Prerequisites

1. **NVIDIA GPU** with CUDA Compute Capability 7.5+ (Turing, Ampere, Ada Lovelace).
2. **NVIDIA CUDA Toolkit** (v12.0+ recommended, tested on v12.9).
3. **CMake** 3.18 or later.
4. **C++17 Compiler:**
   - **Windows:** Visual Studio 2022 (v143) with *Desktop development with C++*.
   - **Linux:** GCC 9+ or Clang 10+.

> **Note for Windows Builds:** Ensure the CUDA Visual Studio MSBuild integration files are installed in Visual Studio's `BuildCustomizations` folder if configuring via MSVC:
> ```powershell
> Copy-Item "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\extras\visual_studio_integration\MSBuildExtensions\*" "C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Microsoft\VC\v170\BuildCustomizations" -Force
> ```

---

## Building

```bash
# Configure build directory
cmake -B build

# Build Release binaries
cmake --build build --config Release
```

---

## Running

The compiled executable is located at `build/Release/montecuda.exe` (Windows) or `build/montecuda` (Linux).

```powershell
# Run with default sample count (1 trillion samples)
.\build\Release\montecuda.exe

# Run with a custom sample count (e.g. 100 million)
.\build\Release\montecuda.exe 100000000
```

---

## Project Structure

```
montecuda/
├── CMakeLists.txt              # CMake build configuration for GPU and CPU targets
├── README.md                   # Project overview, benchmark, and setup instructions
├── .gitignore                  # Build directory and artifact exclusions
└── src/
    ├── pi_monte_carlo.cu       # GPU CUDA kernel + CPU benchmark & comparison output
    └── pi_monte_carlo_cpu.cpp   # Standalone single-threaded CPU implementation
```
