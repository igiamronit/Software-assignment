#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define MAX_ITER 1000
#define EPSILON 1e-9

// Function to perform matrix multiplication: C = A * B
void matrixMultiply(double **A, double **B, double **C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            C[i][j] = 0.0;
            for (int k = 0; k < n; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

// Function to compute QR decomposition using Gram-Schmidt process
void qrDecomposition(double **A, double **Q, double **R, int n) {
    // Initialize Q and R
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            Q[i][j] = (i == j) ? 1.0 : 0.0;
            R[i][j] = 0.0;
        }
    }

    for (int k = 0; k < n; k++) {
        // Compute R[k][k]
        double norm = 0.0;
        for (int i = 0; i < n; i++) {
            norm += A[i][k] * A[i][k];
        }
        R[k][k] = sqrt(norm);

        // Compute Q[:,k]
        for (int i = 0; i < n; i++) {
            Q[i][k] = A[i][k] / R[k][k];
        }

        // Update R[k,j] and A[:,j] for j > k
        for (int j = k + 1; j < n; j++) {
            for (int i = 0; i < n; i++) {
                R[k][j] += Q[i][k] * A[i][j];
            }
            for (int i = 0; i < n; i++) {
                A[i][j] -= Q[i][k] * R[k][j];
            }
        }
    }
}

// Function to compute eigenvalues using the QR algorithm
void qrAlgorithm(double **A, double *eigenvalues, int n) {
    double **Q = malloc(n * sizeof(double *));
    double **R = malloc(n * sizeof(double *));
    double **temp = malloc(n * sizeof(double *));
    for (int i = 0; i < n; i++) {
        Q[i] = malloc(n * sizeof(double));
        R[i] = malloc(n * sizeof(double));
        temp[i] = malloc(n * sizeof(double));
    }

    for (int iter = 0; iter < MAX_ITER; iter++) {
        // Perform QR decomposition
        qrDecomposition(A, Q, R, n);

        // Update A = R * Q
        matrixMultiply(R, Q, temp, n);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                A[i][j] = temp[i][j];
            }
        }

        // Check convergence (off-diagonal elements should be small)
        double offDiagonalNorm = 0.0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (i != j) {
                    offDiagonalNorm += A[i][j] * A[i][j];
                }
            }
        }
        if (sqrt(offDiagonalNorm) < EPSILON) {
            break;
        }
    }

    // Extract eigenvalues (diagonal elements of A)
    for (int i = 0; i < n; i++) {
        eigenvalues[i] = A[i][i];
    }

    // Free allocated memory
    for (int i = 0; i < n; i++) {
        free(Q[i]);
        free(R[i]);
        free(temp[i]);
    }
    free(Q);
    free(R);
    free(temp);
}

int main() {
    int n;
    printf("Enter the size of the matrix: ");
    scanf("%d", &n);

    double **A = malloc(n * sizeof(double *));
    for (int i = 0; i < n; i++) {
        A[i] = malloc(n * sizeof(double));
    }

    printf("Enter the elements of the matrix row-wise:\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            scanf("%lf", &A[i][j]);
        }
    }

    double *eigenvalues = malloc(n * sizeof(double));
    qrAlgorithm(A, eigenvalues, n);

    printf("Eigenvalues:\n");
    for (int i = 0; i < n; i++) {
        printf("%10.6f\n", eigenvalues[i]);
    }

    // Free allocated memory
    for (int i = 0; i < n; i++) {
        free(A[i]);
    }
    free(A);
    free(eigenvalues);

    return 0;
}

