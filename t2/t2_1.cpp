#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <inttypes.h>

int m = 20000, n = 20000;

void matrix_vector_product_omp(double *a, double *b, double *c, int m, int n)
{
    #pragma omp parallel num_threads(40)
    {
        int nthreads = omp_get_num_threads();
        int threadid = omp_get_thread_num();
        int items_per_thread = m / nthreads;
        int lb = threadid * items_per_thread;
        int ub = (threadid == nthreads - 1) ? (m - 1) : (lb + items_per_thread - 1);

        for (int i = lb; i <= ub; i++) {
            c[i] = 0.0;
            for (int j = 0; j < n; j++) c[i] += a[i * n + j] * b[j];
        }
    }
}



void print_matrix(double *a, int m, int n)
{
    printf("Matrix A:\n");
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            printf("%6.1f ", a[i * n + j]);
        }
        printf("\n");
    }
    printf("\n");
}


void print_vector(double *v, int size, const char *name)
{
    printf("Vector %s:\n", name);
    for (int i = 0; i < size; i++) {
        printf("%6.1f ", v[i]);
    }
    printf("\n\n");
}

void run_parallel()
{
    double *a, *b, *c;
    
    #pragma omp parallel num_threads(40)
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();

        if (tid == 0) {
            a = (double*) malloc(sizeof(*a) * m * n);
            b = (double*) malloc(sizeof(*b) * n);
            c = (double*) malloc(sizeof(*c) * m);

            if (!a || !b || !c) {
                printf("Memory allocation failed!\n");
                exit(1);
            }
        }

        #pragma omp barrier // Ждем, пока главный поток выделит память

        #pragma omp for collapse(2)
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                a[i * n + j] = i + j;
            }
        }

        #pragma omp for
        for (int j = 0; j < n; j++) {
            b[j] = j;
        }
    }

    //print_matrix(a, m, n);
    //print_vector(b, n, "B");

    double t = omp_get_wtime();

    matrix_vector_product_omp(a, b, c, m, n);
    t = omp_get_wtime() - t;
    printf("Elapsed time (parallel): %.6f sec.\n", t);

    //print_vector(c, m, "C");

    free(a);
    free(b);
    free(c);
}

int main()
{
    printf("Matrix-vector product (c[m] = a[m, n] * b[n]; m = %d, n = %d)\n", m, n);
    printf("Memory used: %llu MiB\n", (unsigned long long)((m * n + m + n) * sizeof(double)) >> 20);
    
    run_parallel();
    return 0;
}
