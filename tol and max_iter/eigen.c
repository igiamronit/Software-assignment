#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define TOL 1e-16      // Convergence tolerance
#define MAX_ITER 5000   // Maximum QR iterations

// Function to create an n x n zero matrix
double** create_matrix(int n) {
    double **matrix = (double**) malloc(n * sizeof(double*));
    for (int i = 0; i < n; i++) {
        matrix[i] = (double*) calloc(n, sizeof(double));
    }
    return matrix;
}

// Function to free a matrix
void free_matrix(double **matrix, int n) {
    for (int i = 0; i < n; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

// Hessenberg reduction using Householder reflections
void hessenberg_reduction(double **A, int n) {
    for (int k = 0; k < n - 2; k++) {
        double norm = 0.0;
        for (int i = k + 1; i < n; i++) {
            norm += A[i][k] * A[i][k];
        }
        if (norm == 0.0) continue;

        norm = sqrt(norm);
        if (A[k + 1][k] > 0) norm = -norm;

        double u1 = A[k + 1][k] - norm;
        double *u = (double*) calloc(n, sizeof(double));
        u[k + 1] = u1;
        for (int i = k + 2; i < n; i++) {
            u[i] = A[i][k];
        }

        double beta = 0.0;
        for (int i = k + 1; i < n; i++) {
            beta += u[i] * u[i];
        }
        beta = 2.0 / beta;

        // Update matrix A with reflections
        for (int j = k; j < n; j++) {
            double gamma = 0.0;
            for (int i = k + 1; i < n; i++) {
                gamma += u[i] * A[i][j];
            }
            gamma *= beta;
            for (int i = k + 1; i < n; i++) {
                A[i][j] -= gamma * u[i];
            }
        }

        for (int i = 0; i < n; i++) {
            double gamma = 0.0;
            for (int j = k + 1; j < n; j++) {
                gamma += u[j] * A[i][j];
            }
            gamma *= beta;
            for (int j = k + 1; j < n; j++) {
                A[i][j] -= gamma * u[j];
            }
        }

        A[k + 1][k] = norm;
        for (int i = k + 2; i < n; i++) {
            A[i][k] = 0.0;
        }

        free(u);
    }
}

// QR decomposition using Givens rotations
void qr_decomposition(double **A, double **Q, double **R, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            Q[i][j] = (i == j) ? 1.0 : 0.0;
            R[i][j] = A[i][j];
        }
    }

    for (int k = 0; k < n - 1; k++) {
        for (int i = k + 1; i < n; i++) {
            if (fabs(R[i][k]) < TOL) continue;

            double r = hypot(R[k][k], R[i][k]);
            double c = R[k][k] / r;
            double s = -R[i][k] / r;

            // Apply Givens rotation
            for (int j = 0; j < n; j++) {
                double temp_kj = c * R[k][j] - s * R[i][j];
                double temp_ij = s * R[k][j] + c * R[i][j];
                R[k][j] = temp_kj;
                R[i][j] = temp_ij;

                temp_kj = c * Q[k][j] - s * Q[i][j];
                temp_ij = s * Q[k][j] + c * Q[i][j];
                Q[k][j] = temp_kj;
                Q[i][j] = temp_ij;
            }
        }
    }
}

// QR iteration to find eigenvalues, with iteration count
void qr_algorithm(double **A, int n, double *eigenvalues) {
    double **Q = create_matrix(n);
    double **R = create_matrix(n);
    int iterations = 0;

    for (iterations = 0; iterations < MAX_ITER; iterations++) {
        qr_decomposition(A, Q, R, n);

        // Form new A as R * Q
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                A[i][j] = 0.0;
                for (int k = 0; k < n; k++) {
                    A[i][j] += R[i][k] * Q[k][j];
                }
            }
        }

        // Check for convergence
        int converged = 1;
        for (int i = 1; i < n && converged; i++) {
            if (fabs(A[i][i - 1]) > TOL) {
                converged = 0;
            }
        }
        if (converged) break;
    }

    // Store the eigenvalues
    for (int i = 0; i < n; i++) {
        eigenvalues[i] = A[i][i];
    }

    printf("Number of iterations: %d\n", iterations);  // Output the iteration count

    free_matrix(Q, n);
    free_matrix(R, n);
}

// Function to read matrix input
void read_matrix(double **A, int n) {
    printf("Enter the elements of the %dx%d matrix:\n", n, n);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            scanf("%lf", &A[i][j]);
        }
    }
}

// Main function
int main() {
    int n;
    printf("Enter the size of the matrix (n x n): ");
    scanf("%d", &n);

    double **A = create_matrix(n);
    double *eigenvalues = (double*) malloc(n * sizeof(double));

    read_matrix(A, n);

    hessenberg_reduction(A, n);
    qr_algorithm(A, n, eigenvalues);

    printf("Eigenvalues:\n");
    for (int i = 0; i < n; i++) {
        printf("%lf\n", eigenvalues[i]);
    }

    free_matrix(A, n);
    free(eigenvalues);

    return 0;
}

