#include <iostream>
#include <vector>
#include <iomanip>
#include <cuda_runtime.h> // for cuda functions
#include <time.h> // for time(NULL)
#include <cmath> // for std::abs and std::acos
#include <chrono>
#include <random>
#include <stdexcept> // for std::invalid_argument
#include <string> // for std::stoull
#include <curand_kernel.h> // for curand functions

#define CUDA_CHECK(err) { \
    cudaError_t err_ = (err); \
    if (err_ != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err_)); \
        exit(EXIT_FAILURE); \
    } \
}

#define THREADS_PER_BLOCK 256

__global__ void estimate_pi_kernel(unsigned long long seed, unsigned long long samples_per_thread, unsigned long long* block_results) {
    // shared memory for reduction within a block
    __shared__ unsigned long long shared_counts[THREADS_PER_BLOCK];

    // get thread and block indices
    unsigned int tid = threadIdx.x;
    unsigned int block_id = blockIdx.x;
    unsigned int global_thread_id = block_id * blockDim.x + tid;

    // Use unsigned long long for the local count to prevent overflow if samples_per_thread
    // exceeds the limit of unsigned int (~4.29 billion).
    unsigned long long local_count = 0;

    // initialize random number generator state for each thread
    curandState_t state;
    curand_init(seed + global_thread_id, 0, 0, &state);

    // each thread generates its assigned number of random points
    for (unsigned long long i = 0; i < samples_per_thread; ++i) {
        // generate random x and y coordinates between 0.0 and 1.0
        float x = curand_uniform(&state); // random x coordinate
        float y = curand_uniform(&state); // random y coordinate

        // check if the point is within the unit circle
        if (x * x + y * y <= 1.0f) {
            local_count++;
        }
    }

    // store local count in shared memory
    shared_counts[tid] = local_count;
    __syncthreads(); // synchronize threads within the block

    // perform parallel reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_counts[tid] += shared_counts[tid + s];
        }
        __syncthreads(); // synchronize after each reduction step
    }

    // only the first thread in the block writes the total count to global memory
    if (tid == 0) {
        block_results[block_id] = shared_counts[0];
    }
}

struct BenchmarkResult {
    unsigned long long samples;
    unsigned long long in_circle;
    double time_ms;
    double pi_estimate;
    double error;
};

BenchmarkResult run_gpu_simulation(unsigned long long total_samples) {
    // get device properties
    int device_id;
    CUDA_CHECK(cudaGetDevice(&device_id));
    cudaDeviceProp props; 
    CUDA_CHECK(cudaGetDeviceProperties(&props, device_id));

    // determine number of blocks based on multiprocessor count
    int num_blocks = props.multiProcessorCount * 32; 

    // calculate total threads and samples per thread
    unsigned long long total_threads = (unsigned long long)num_blocks * THREADS_PER_BLOCK;
    // use ceiling division to ensure at least total_samples are generated
    unsigned long long samples_per_thread = (total_samples + total_threads - 1) / total_threads; 
    unsigned long long actual_total_samples = samples_per_thread * total_threads;

    std::cout << "============================================================" << std::endl;
    std::cout << "  GPU (CUDA) -- Parallelized" << std::endl;
    std::cout << "============================================================" << std::endl;
    std::cout << "GPU:                     " << props.name << std::endl;
    std::cout << "Threads per block:       " << THREADS_PER_BLOCK << std::endl;
    std::cout << "Number of blocks:        " << num_blocks << std::endl;
    std::cout << "Samples per thread:      " << samples_per_thread << std::endl;
    std::cout << "Total samples requested: " << total_samples << std::endl;
    std::cout << "Total samples actual:    " << actual_total_samples << std::endl;

    // host and device memory allocation for block results
    std::vector<unsigned long long> h_block_results(num_blocks);
    unsigned long long* d_block_results;
    CUDA_CHECK(cudaMalloc(&d_block_results, num_blocks * sizeof(unsigned long long)));

    // cuda events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // seed for random number generation
    unsigned long long seed = (unsigned long long)time(NULL);

    // record start time, launch kernel, record stop time
    CUDA_CHECK(cudaEventRecord(start));
    estimate_pi_kernel<<<num_blocks, THREADS_PER_BLOCK>>>(seed, samples_per_thread, d_block_results);
    CUDA_CHECK(cudaGetLastError()); // check for kernel launch errors
    CUDA_CHECK(cudaEventRecord(stop));

    // copy results from device to host
    CUDA_CHECK(cudaMemcpy(h_block_results.data(), d_block_results, num_blocks * sizeof(unsigned long long), cudaMemcpyDeviceToHost));

    // sum up all block results
    unsigned long long total_in_circle = 0;
    for (int i = 0; i < num_blocks; ++i) {
        total_in_circle += h_block_results[i];
    }

    // calculate elapsed time
    float milliseconds = 0;
    CUDA_CHECK(cudaEventSynchronize(stop)); // wait for stop event to complete
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));

    // clean up cuda resources
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_block_results));

    double actual_pi = std::acos(-1.0);
    double pi_estimate = 4.0 * static_cast<double>(total_in_circle) / static_cast<double>(actual_total_samples);
    double error = std::abs(pi_estimate - actual_pi);

    std::cout << std::fixed << std::setprecision(12);
    std::cout << "\nPoints in circle:      " << total_in_circle << std::endl;
    std::cout << "Kernel execution time: " << milliseconds << " ms" << std::endl;
    std::cout << "Estimated Pi =         " << pi_estimate << std::endl;
    std::cout << "Actual Pi    =         " << actual_pi << std::endl;
    std::cout << "Error        =         " << error << std::endl;

    return {actual_total_samples, total_in_circle, (double)milliseconds, pi_estimate, error};
}

BenchmarkResult run_cpu_simulation(unsigned long long total_samples) {
    std::cout << "\n============================================================" << std::endl;
    std::cout << "  CPU -- Single-threaded (No parallelization)" << std::endl;
    std::cout << "============================================================" << std::endl;
    std::cout << "Total samples:           " << total_samples << std::endl;

    std::mt19937_64 rng(static_cast<unsigned long long>(std::time(nullptr)));
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    unsigned long long total_in_circle = 0;

    auto start = std::chrono::high_resolution_clock::now();

    for (unsigned long long i = 0; i < total_samples; ++i) {
        float x = dist(rng);
        float y = dist(rng);

        if (x * x + y * y <= 1.0f) {
            total_in_circle++;
        }
    }

    auto stop = std::chrono::high_resolution_clock::now();
    double milliseconds = std::chrono::duration<double, std::milli>(stop - start).count();

    double actual_pi = std::acos(-1.0);
    double pi_estimate = 4.0 * static_cast<double>(total_in_circle) / static_cast<double>(total_samples);
    double error = std::abs(pi_estimate - actual_pi);

    std::cout << std::fixed << std::setprecision(12);
    std::cout << "\nPoints in circle:      " << total_in_circle << std::endl;
    std::cout << "Execution time:        " << milliseconds << " ms" << std::endl;
    std::cout << "Estimated Pi =         " << pi_estimate << std::endl;
    std::cout << "Actual Pi    =         " << actual_pi << std::endl;
    std::cout << "Error        =         " << error << std::endl;

    return {total_samples, total_in_circle, milliseconds, pi_estimate, error};
}

void print_comparison(const BenchmarkResult& gpu, const BenchmarkResult& cpu) {
    double speedup = (gpu.time_ms > 0.0) ? (cpu.time_ms / gpu.time_ms) : 0.0;

    std::cout << "\n============================================================" << std::endl;
    std::cout << "  COMPARISON: GPU vs CPU" << std::endl;
    std::cout << "============================================================" << std::endl;
    std::cout << std::left << std::setw(24) << "Metric" 
              << std::right << std::setw(18) << "GPU (CUDA)" 
              << std::setw(18) << "CPU (Single)" << std::endl;
    std::cout << "  --------------------------------------------------------" << std::endl;
    std::cout << std::left << std::setw(24) << "Samples"
              << std::right << std::setw(18) << gpu.samples
              << std::setw(18) << cpu.samples << std::endl;
    std::cout << std::left << std::setw(24) << "Time (ms)"
              << std::right << std::fixed << std::setprecision(2) << std::setw(18) << gpu.time_ms
              << std::setw(18) << cpu.time_ms << std::endl;
    std::cout << std::left << std::setw(24) << "Estimated Pi"
              << std::right << std::fixed << std::setprecision(10) << std::setw(18) << gpu.pi_estimate
              << std::setw(18) << cpu.pi_estimate << std::endl;
    std::cout << std::left << std::setw(24) << "Error"
              << std::right << std::fixed << std::setprecision(10) << std::setw(18) << gpu.error
              << std::setw(18) << cpu.error << std::endl;
    std::cout << std::left << std::setw(24) << "Points in circle"
              << std::right << std::setw(18) << gpu.in_circle
              << std::setw(18) << cpu.in_circle << std::endl;

    std::cout << "\n  >>> GPU speedup: " << std::fixed << std::setprecision(2) << speedup << "x faster than CPU <<<\n" << std::endl;
}

int main(int argc, char** argv) {
    // 1 trillion samples default: 1,000,000,000,000 ULL
    unsigned long long total_samples = 1000000000000ULL;
    if (argc > 1) {
        try {
            total_samples = std::stoull(argv[1]);
        } catch (const std::invalid_argument& ia) {
            std::cerr << "Invalid argument: " << argv[1] << ". Please provide a number." << std::endl;
            return 1;
        }
    }

    // Run GPU simulation
    BenchmarkResult gpu_res = run_gpu_simulation(total_samples);

    // Run CPU simulation directly on all samples without extrapolation
    BenchmarkResult cpu_res = run_cpu_simulation(total_samples);

    print_comparison(gpu_res, cpu_res);

    return 0;
}