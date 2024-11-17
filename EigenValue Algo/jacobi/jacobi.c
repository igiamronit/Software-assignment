#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#define MAX_ITER 1000
#define EPSILON 1e-9

void printMatrix(double **matrix, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%10.6f ", matrix[i][j]);
        }
        printf("\n");
    }
}

void jacobi(double **A, double *eigenvalues, int n) {
    double **V = malloc(n * sizeof(double *));
    for (int i = 0; i < n; i++) {
        V[i] = malloc(n * sizeof(double));
        for (int j = 0; j < n; j++) {
            V[i][j] = (i == j) ? 1.0 : 0.0; // Initialize V as identity matrix
        }
    }

    for (int iter = 0; iter < MAX_ITER; iter++) {
        // Find the largest off-diagonal element
        int p, q;
        double max = 0.0;
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                if (fabs(A[i][j]) > max) {
                    max = fabs(A[i][j]);
                    p = i;
                    q = j;
                }
            }
        }

        if (max < EPSILON) {
            break; // Convergence
        }

        // Calculate the rotation
        double theta = 0.5 * atan2(2 * A[p][q], A[q][q] - A[p][p]);
        double c = cos(theta);
        double s = sin(theta);

        // Update matrix A
        for (int i = 0; i < n; i++) {
            if (i != p && i != q) {
                double aip = A[i][p];
                double aiq = A[i][q];
                A[i][p] = A[p][i] = c * aip - s * aiq;
                A[i][q] = A[q][i] = c * aiq + s * aip;
            }
        }

        double app = A[p][p];
        double aqq = A[q][q];
        double apq = A[p][q];
        A[p][p] = c * c * app + s * s * aqq - 2 * s * c * apq;
        A[q][q] = s * s * app + c * c * aqq + 2 * s * c * apq;
        A[p][q] = A[q][p] = 0.0;

        // Update eigenvector matrix V
        for (int i = 0; i < n; i++) {
            double vip = V[i][p];
            double viq = V[i][q];
            V[i][p] = c * vip - s * viq;
            V[i][q] = s * vip + c * viq;
        }
    }

    // Extract eigenvalues
    for (int i = 0; i < n; i++) {
        eigenvalues[i] = A[i][i];
    }

    // Free allocated memory
    for (int i = 0; i < n; i++) {
        free(V[i]);
    }
    free(V);
}

int main() {
    int n;
    printf("Enter the size of the matrix: ");
    scanf("%d", &n);

    double **A = malloc(n * sizeof(double *));
    for (int i = 0; i < n; i++) {
        A[i] = malloc(n * sizeof(double));
    }

    printf("Enter the elements of the symmetric matrix row-wise:\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            scanf("%lf", &A[i][j]);
        }
    }

    double *eigenvalues = malloc(n * sizeof(double));
    jacobi(A, eigenvalues, n);

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

