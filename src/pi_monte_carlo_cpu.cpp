#include <iostream>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <random>
#include <stdexcept>
#include <string>

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

    std::cout << "CPU: Single-threaded (no parallelization)" << std::endl;
    std::cout << "Total samples: " << total_samples << std::endl;

    // seed the random number generator
    std::mt19937_64 rng(static_cast<unsigned long long>(std::time(nullptr)));
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    unsigned long long total_in_circle = 0;

    // start timing
    auto start = std::chrono::high_resolution_clock::now();

    // generate random points and count those inside the unit circle
    for (unsigned long long i = 0; i < total_samples; ++i) {
        float x = dist(rng);
        float y = dist(rng);

        if (x * x + y * y <= 1.0f) {
            total_in_circle++;
        }
    }

    // stop timing
    auto stop = std::chrono::high_resolution_clock::now();
    double milliseconds = std::chrono::duration<double, std::milli>(stop - start).count();

    // calculate pi estimate and error
    double pi_estimate = 4.0 * static_cast<double>(total_in_circle) / static_cast<double>(total_samples);
    double actual_pi = std::acos(-1.0);

    // print results
    std::cout << std::fixed << std::setprecision(12);
    std::cout << "\nPoints in circle: " << total_in_circle << std::endl;
    std::cout << "Execution time:   " << milliseconds << " ms" << std::endl;
    std::cout << "Estimated Pi =    " << pi_estimate << std::endl;
    std::cout << "Actual Pi    =    " << actual_pi << std::endl;
    std::cout << "Error        =    " << std::abs(pi_estimate - actual_pi) << std::endl;

    return 0;
}
