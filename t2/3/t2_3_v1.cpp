#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <time.h>
#include <omp.h>

using namespace std;

const double EPSILON = 1e-5; 
const double TAU = 0.000001;

using Vector = vector<double>;
using Matrix = vector<Vector>;


double norm(const Vector &v, int n_threads) {
    double sum = 0.0;

    #pragma omp parallel for reduction(+:sum) num_threads(n_threads)
    for (size_t i = 0; i < v.size(); ++i) {
        sum += v[i] * v[i];
    }

    return sqrt(sum);
}

Vector simpleIterationMethod(const Matrix &A, const Vector &b, int n_threads) {
    int n = A.size();
    Vector x(n, 0.0);
    Vector Ax(n);

    while (true) {

        int n = A.size();
        Vector result(n, 0.0);

        #pragma omp parallel for num_threads(n_threads)
        for (int i = 0; i < n; ++i) {
        
            for (int j = 0; j < n; ++j) {
                result[i] += A[i][j] * x[j];
            }
        }
        Ax = result;

        Vector r(n);

        #pragma omp parallel for num_threads(n_threads)
        for (int i = 0; i < n; ++i) {
            r[i] = Ax[i] - b[i];
        }
        
        
        if (norm(r, n_threads) / norm(b, n_threads) < EPSILON) {
            break;
        }

        #pragma omp parallel for num_threads(n_threads)
        for (int i = 0; i < n; ++i) {
            x[i] -= TAU * r[i];
        }
        //cout << x[0] << " ";
    }

    return x;
}

int main() {
    int N;
    cout << "Enter the number of equations (N): ";
    cin >> N;

    std::ofstream out;
    out.open("Out_v1.txt");

    for(int n_threads = 2; n_threads<=80; n_threads++){
    Matrix A;
    Vector b;
    
    double t = omp_get_wtime();

    A.assign(N, Vector(N, 1.0));

    #pragma omp parallel for num_threads(n_threads)
    for (int i = 0; i < N; ++i) {
        A[i][i] = 2.0; 
    }
    b.assign(N, N + 1); 
    

    Vector solution = simpleIterationMethod(A, b, n_threads);

    t = omp_get_wtime() - t;
/*
    cout << "Solution: ";

    for (double val : solution) {
        cout << val << " ";
    }
    cout << endl;
*/
    printf("n_threads: %d Execution time (parallel): %.6f\n", n_threads, t);
    out << n_threads << "   " << t << "   " << 82.108765/t << "\n";
    }
    out.close();
    cout << "File has been written" << std::endl; 
    return 0;
}
