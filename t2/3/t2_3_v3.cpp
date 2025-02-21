#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <time.h>
#include <omp.h>

#define N_THREADS 40

using namespace std;

const double EPSILON = 1e-5; 
const double TAU = 0.000001;

using Vector = vector<double>;
using Matrix = vector<Vector>;

double norm(const Vector &v) {
    double sum = 0.0;
    for (double val : v) {
        sum += val * val;
    }
    return sqrt(sum);
}

Vector simpleIterationMethod(const Matrix &A, const Vector &b, int n_threads, const string& schedule_type, int chunk_size) {
    int n = A.size();
    Vector x(n, 0.0);
    Vector Ax(n);
    Vector r(n);
    bool stop = false;

    #pragma omp parallel num_threads(n_threads) 
    {
        while (true) {
            #pragma omp single
            Ax.assign(n, 0.0);

            if (schedule_type == "static") {
                #pragma omp for schedule(static, chunk_size) nowait
                for (int i = 0; i < n; ++i) {
                    for (int j = 0; j < n; ++j) {
                        Ax[i] += A[i][j] * x[j];
                    }
                }
            } else if (schedule_type == "dynamic") {
                #pragma omp for schedule(dynamic, chunk_size) nowait
                for (int i = 0; i < n; ++i) {
                    for (int j = 0; j < n; ++j) {
                        Ax[i] += A[i][j] * x[j];
                    }
                }
            } else if (schedule_type == "guided") {
                #pragma omp for schedule(guided, chunk_size) nowait
                for (int i = 0; i < n; ++i) {
                    for (int j = 0; j < n; ++j) {
                        Ax[i] += A[i][j] * x[j];
                    }
                }
            } else if (schedule_type == "auto") {
                #pragma omp for schedule(guided, chunk_size) nowait
                for (int i = 0; i < n; ++i) {
                    for (int j = 0; j < n; ++j) {
                        Ax[i] += A[i][j] * x[j];
                    }
                }
            }

            #pragma omp single
            r.assign(n, 0.0);

            #pragma omp for nowait
            for (int i = 0; i < n; ++i) {
                r[i] = Ax[i] - b[i];
            }

            #pragma omp single
            {
                if (norm(r) / norm(b) < EPSILON) {
                    stop = true;
                }
            }

            if (stop) break;

            #pragma omp for nowait
            for (int i = 0; i < n; ++i) {
                x[i] -= TAU * r[i];
            }
        }
    }

    return x;
}

int main() {
    int N;
    cout << "Enter the number of equations (N): ";
    cin >> N;

    vector<string> schedules = {"static", "dynamic", "guided", "auto"};
    vector<int> chunk_sizes = {1, 4, 8, 16};

    std::ofstream out("Out_v3.txt");
    out << "# threads  schedule  chunk_size  time  speedup\n"; 

    for (int n_threads = 2; n_threads <= 80; n_threads += 2) {
        for (const auto& schedule : schedules) {
            for (int chunk_size : chunk_sizes) {
                Matrix A(N, Vector(N, 1.0));
                Vector b(N, N + 1);

                #pragma omp parallel for num_threads(n_threads)
                for (int i = 0; i < N; ++i) {
                    A[i][i] = 2.0;
                }

                double t = omp_get_wtime();
                Vector solution = simpleIterationMethod(A, b, n_threads, schedule, chunk_size);
                t = omp_get_wtime() - t;

                printf("Threads: %d | Schedule: %s | Chunk: %d | Time: %.6f sec\n", n_threads, schedule.c_str(), chunk_size, t);
                out << n_threads << "\t" << schedule << "\t" << chunk_size << "\t" << t << "\t" << 82.108765 / t << "\n";
            }
        }
    }

    out.close();
    cout << "File has been written" << std::endl;

    return 0;
}
