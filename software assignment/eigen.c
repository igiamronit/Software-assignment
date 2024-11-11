#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define TOL 1e-10  // Tolerance for convergence
#define MAX_ITER 1000  // Maximum iterations

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

// Function to copy matrix B into matrix A
void copy_matrix(double **A, double **B, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A[i][j] = B[i][j];
        }
    }
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
        A[k + 1][k] = norm;
        for (int i = k + 2; i < n; i++) {
            A[i][k] /= u1;
        }

        for (int j = k + 1; j < n; j++) {
            double beta = A[k + 1][j];
            for (int i = k + 2; i < n; i++) {
                beta += A[i][k] * A[i][j];
            }
            beta /= u1 * norm;

            A[k + 1][j] -= beta * u1;
            for (int i = k + 2; i < n; i++) {
                A[i][j] -= beta * A[i][k];
            }
        }

        for (int j = k + 1; j < n; j++) {
            A[j][k] = 0.0;
        }
    }
}

// QR decomposition with Gram-Schmidt orthogonalization
void qr_decomposition(double **A, double **Q, double **R, int n) {
    for (int j = 0; j < n; j++) {
        for (int i = 0; i < n; i++) {
            Q[i][j] = A[i][j];
        }

        for (int i = 0; i < j; i++) {
            R[i][j] = 0.0;
            for (int k = 0; k < n; k++) {
                R[i][j] += Q[k][i] * Q[k][j];
            }
            for (int k = 0; k < n; k++) {
                Q[k][j] -= R[i][j] * Q[k][i];
            }
        }

        R[j][j] = 0.0;
        for (int k = 0; k < n; k++) {
            R[j][j] += Q[k][j] * Q[k][j];
        }
        R[j][j] = sqrt(R[j][j]);
        for (int k = 0; k < n; k++) {
            Q[k][j] /= R[j][j];
        }
    }
}

// QR iteration to find eigenvalues
void qr_algorithm(double **A, int n) {
    double **Q = create_matrix(n);
    double **R = create_matrix(n);
    double **temp = create_matrix(n);
    
    // QR Iteration
    for (int iter = 0; iter < MAX_ITER; iter++) {
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

    // Print eigenvalues (diagonal elements of converged A)
    printf("Eigenvalues:\n");
    for (int i = 0; i < n; i++) {
        printf("%lf\n", A[i][i]);
    }
    
    // Free allocated matrices
    free_matrix(Q, n);
    free_matrix(R, n);
    free_matrix(temp, n);
}

// Main function
int main() {
    int n;
    printf("Enter matrix size: ");
    scanf("%d", &n);

    double **A = create_matrix(n);
    
    printf("Enter matrix elements:\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            scanf("%lf", &A[i][j]);
        }
    }

    // Step 1: Hessenberg reduction
    hessenberg_reduction(A, n);

    // Step 2: Apply QR algorithm
    qr_algorithm(A, n);

    // Free the matrix
    free_matrix(A, n);
    
    return 0;
}

