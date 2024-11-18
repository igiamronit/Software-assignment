#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define MAX_ITER 1000
#define EPSILON 1e-9

// Function to print a matrix
void printMatrix(double **M, int n, const char *label) {
    printf("%s:\n", label);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%10.6f ", M[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

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

// Function to transpose a matrix
void transpose(double **A, double **T, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            T[j][i] = A[i][j];
        }
    }
}

// Function to perform Hessenberg reduction
void hessenbergReduction(double **A, double **H, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            H[i][j] = A[i][j];
        }
    }

    for (int k = 0; k < n - 2; k++) {
        double norm = 0.0;
        for (int i = k + 1; i < n; i++) {
            norm += H[i][k] * H[i][k];
        }
        norm = sqrt(norm);
        if (norm < EPSILON) continue;

        double *u = malloc(n * sizeof(double));
        for (int i = 0; i < n; i++) u[i] = 0.0;

        u[k + 1] = H[k + 1][k] + (H[k + 1][k] < 0 ? -norm : norm);
        for (int i = k + 2; i < n; i++) {
            u[i] = H[i][k];
        }

        double u_norm = 0.0;
        for (int i = k + 1; i < n; i++) {
            u_norm += u[i] * u[i];
        }
        u_norm = sqrt(u_norm);
        if (u_norm < EPSILON) {
            free(u);
            continue;
        }
        for (int i = k + 1; i < n; i++) {
            u[i] /= u_norm;
        }

        for (int j = k; j < n; j++) {
            double dot = 0.0;
            for (int i = k + 1; i < n; i++) {
                dot += u[i] * H[i][j];
            }
            for (int i = k + 1; i < n; i++) {
                H[i][j] -= 2 * u[i] * dot;
            }
        }

        for (int i = 0; i < n; i++) {
            double dot = 0.0;
            for (int j = k + 1; j < n; j++) {
                dot += u[j] * H[i][j];
            }
            for (int j = k + 1; j < n; j++) {
                H[i][j] -= 2 * u[j] * dot;
            }
        }

        free(u);
    }

    printMatrix(H, n, "Hessenberg Reduced Matrix");
}

// Function to perform QR decomposition on a Hessenberg matrix
void qrDecompositionHessenberg(double **H, double **Q, double **R, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            Q[i][j] = (i == j) ? 1.0 : 0.0;
        }
    }

    for (int k = 0; k < n - 1; k++) {
        double x = H[k][k];
        double y = H[k + 1][k];
        double r = sqrt(x * x + y * y);

        if (r < EPSILON) continue;

        double c = x / r;
        double s = -y / r;

        for (int j = k; j < n; j++) {
            double temp1 = c * H[k][j] - s * H[k + 1][j];
            double temp2 = s * H[k][j] + c * H[k + 1][j];
            H[k][j] = temp1;
            H[k + 1][j] = temp2;
        }

        for (int i = 0; i < n; i++) {
            double temp1 = c * Q[i][k] - s * Q[i][k + 1];
            double temp2 = s * Q[i][k] + c * Q[i][k + 1];
            Q[i][k] = temp1;
            Q[i][k + 1] = temp2;
        }
    }

    transpose(Q, R, n);
    matrixMultiply(R, H, R, n);
    printMatrix(H, n, "H Matrix after QR Decomposition");
}

// QR Algorithm with Hessenberg reduction
void qrAlgorithmHessenberg(double **A, double *eigenvalues, int n) {
    double **H = malloc(n * sizeof(double *));
    double **Q = malloc(n * sizeof(double *));
    double **R = malloc(n * sizeof(double *));
    double **temp = malloc(n * sizeof(double *));
    for (int i = 0; i < n; i++) {
        H[i] = malloc(n * sizeof(double));
        Q[i] = malloc(n * sizeof(double));
        R[i] = malloc(n * sizeof(double));
        temp[i] = malloc(n * sizeof(double));
    }

    hessenbergReduction(A, H, n);

    for (int iter = 0; iter < MAX_ITER; iter++) {
        qrDecompositionHessenberg(H, Q, R, n);
        matrixMultiply(R, Q, temp, n);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                H[i][j] = temp[i][j];
            }
        }

        double offDiagonalNorm = 0.0;
        for (int i = 0; i < n - 1; i++) {
            offDiagonalNorm += H[i + 1][i] * H[i + 1][i];
        }
        if (sqrt(offDiagonalNorm) < EPSILON) break;
    }

    for (int i = 0; i < n; i++) {
        eigenvalues[i] = H[i][i];
    }

    for (int i = 0; i < n; i++) {
        free(H[i]);
        free(Q[i]);
        free(R[i]);
        free(temp[i]);
    }
    free(H);
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
    qrAlgorithmHessenberg(A, eigenvalues, n);

    printf("Eigenvalues:\n");
    for (int i = 0; i < n; i++) {
        printf("%10.6f\n", eigenvalues[i]);
    }

    for (int i = 0; i < n; i++) {
        free(A[i]);
    }
    free(A);
    free(eigenvalues);

    return 0;
}
