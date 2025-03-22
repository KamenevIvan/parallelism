#include <iostream>
#include <vector>
#include <thread>
#include <chrono>
#include <cstdlib>

int m = 40000, n = 40000;

void matrix_vector_product(const std::vector<double>& a, const std::vector<double>& b, std::vector<double>& c, int start, int end) {
    for (int i = start; i < end; ++i) {
        c[i] = 0.0;
        for (int j = 0; j < n; ++j) {
            c[i] += a[i * n + j] * b[j];
        }
    }
}

void initialize_matrix(std::vector<double>& a, int start, int end) {
    for (int i = start; i < end; ++i) {
        for (int j = 0; j < n; ++j) {
            a[i * n + j] = i + j;
        }
    }
}

void initialize_vector(std::vector<double>& b, int start, int end) {
    for (int j = start; j < end; ++j) {
        b[j] = j;
    }
}

void run_parallel(int num_threads) {
    std::vector<double> a(m * n);
    std::vector<double> b(n);
    std::vector<double> c(m);

    std::vector<std::thread> threads;

    int chunk_size = m / num_threads;
    for (int i = 0; i < num_threads; ++i) {
        int start = i * chunk_size;
        int end = (i == num_threads - 1) ? m : start + chunk_size;
        threads.emplace_back(initialize_matrix, std::ref(a), start, end);
    }
    for (auto& t : threads) {
        t.join();
    }

    threads.clear();
    chunk_size = n / num_threads;
    for (int i = 0; i < num_threads; ++i) {
        int start = i * chunk_size;
        int end = (i == num_threads - 1) ? n : start + chunk_size;
        threads.emplace_back(initialize_vector, std::ref(b), start, end);
    }
    for (auto& t : threads) {
        t.join();
    }

    auto start_time = std::chrono::high_resolution_clock::now();

    threads.clear();
    chunk_size = m / num_threads;
    for (int i = 0; i < num_threads; ++i) {
        int start = i * chunk_size;
        int end = (i == num_threads - 1) ? m : start + chunk_size;
        threads.emplace_back(matrix_vector_product, std::cref(a), std::cref(b), std::ref(c), start, end);
    }
    for (auto& t : threads) {
        t.join();
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    std::cout << "Elapsed time with " << num_threads << " threads: " << elapsed.count() << " sec.\n";
}

int main() {
    std::cout << "Matrix-vector product (c[m] = a[m, n] * b[n]; m = " << m << ", n = " << n << ")\n";
    std::cout << "Memory used: " << ((m * n + m + n) * sizeof(double)) / (1024 * 1024) << " MiB\n";

    for (int num_threads : {1,2,4,7,8,16,20,40}) {
        run_parallel(num_threads);
    }

    return 0;
}