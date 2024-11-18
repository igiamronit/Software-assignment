#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <time.h> // For measuring execution time

// Function prototypes
double **allocateMatrix(int n);
void freeMatrix(double **matrix, int n);
void qrDecomposition(double **H, double **Q, double **R, int n);
void hessenbergReduction(double **A, int n);
int qrIteration(double **H, int n, int maxIter, double tol, double complex *eigenvalues);

// Helper functions
double **allocateMatrix(int n) {
    double **matrix = (double **)malloc(n * sizeof(double *));
    for (int i = 0; i < n; i++) {
        matrix[i] = (double *)malloc(n * sizeof(double));
    }
    return matrix;
}

void freeMatrix(double **matrix, int n) {
    for (int i = 0; i < n; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

void qrDecomposition(double **H, double **Q, double **R, int n) {
    // Initialize Q as the identity matrix
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            Q[i][j] = (i == j) ? 1.0 : 0.0;
        }
    }

    for (int k = 0; k < n - 1; k++) {
        double norm = 0.0;
        for (int i = k; i < n; i++) {
            norm += H[i][k] * H[i][k];
        }
        norm = sqrt(norm);

        double alpha = (H[k][k] > 0) ? -norm : norm;
        double r = sqrt(0.5 * (alpha * alpha - H[k][k] * alpha));
        double *v = (double *)malloc(n * sizeof(double));

        for (int i = 0; i < n; i++) {
            v[i] = 0.0;
        }
        v[k] = (H[k][k] - alpha) / (2.0 * r);
        for (int i = k + 1; i < n; i++) {
            v[i] = H[i][k] / (2.0 * r);
        }

        // Update H = (I - 2vv^T) H
        for (int i = k; i < n; i++) {
            double sum = 0.0;
            for (int j = k; j < n; j++) {
                sum += v[j] * H[j][i];
            }
            for (int j = k; j < n; j++) {
                H[j][i] -= 2.0 * v[j] * sum;
            }
        }

        // Update Q = Q (I - 2vv^T)
        for (int i = 0; i < n; i++) {
            double sum = 0.0;
            for (int j = k; j < n; j++) {
                sum += v[j] * Q[i][j];
            }
            for (int j = k; j < n; j++) {
                Q[i][j] -= 2.0 * v[j] * sum;
            }
        }

        free(v);
    }

    // Extract R
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            R[i][j] = (i <= j) ? H[i][j] : 0.0;
        }
    }
}

void hessenbergReduction(double **A, int n) {
    for (int k = 0; k < n - 2; k++) {
        double norm = 0.0;
        for (int i = k + 1; i < n; i++) {
            norm += A[i][k] * A[i][k];
        }
        norm = sqrt(norm);

        double alpha = (A[k + 1][k] > 0) ? -norm : norm;
        double r = sqrt(0.5 * (alpha * alpha - A[k + 1][k] * alpha));
        double *v = (double *)malloc(n * sizeof(double));

        for (int i = 0; i < n; i++) {
            v[i] = 0.0;
        }
        v[k + 1] = (A[k + 1][k] - alpha) / (2.0 * r);
        for (int i = k + 2; i < n; i++) {
            v[i] = A[i][k] / (2.0 * r);
        }

        // Update A = (I - 2vv^T) A (I - 2vv^T)
        for (int i = k; i < n; i++) {
            double sum = 0.0;
            for (int j = k + 1; j < n; j++) {
                sum += v[j] * A[j][i];
            }
            for (int j = k + 1; j < n; j++) {
                A[j][i] -= 2.0 * v[j] * sum;
            }
        }

        for (int i = 0; i < n; i++) {
            double sum = 0.0;
            for (int j = k + 1; j < n; j++) {
                sum += v[j] * A[i][j];
            }
            for (int j = k + 1; j < n; j++) {
                A[i][j] -= 2.0 * v[j] * sum;
            }
        }

        free(v);
    }
}

int qrIteration(double **H, int n, int maxIter, double tol, double complex *eigenvalues) {
    int iter = 0;
    clock_t start, end;
    start = clock();

    for (iter = 0; iter < maxIter; iter++) {
        // QR decomposition
        double **Q = allocateMatrix(n);
        double **R = allocateMatrix(n);
        qrDecomposition(H, Q, R, n);

        // Update H = R * Q
        double **temp = allocateMatrix(n);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                temp[i][j] = 0.0;
                for (int k = 0; k < n; k++) {
                    temp[i][j] += R[i][k] * Q[k][j];
                }
            }
        }

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                H[i][j] = temp[i][j];
            }
        }

        freeMatrix(Q, n);
        freeMatrix(R, n);
        freeMatrix(temp, n);

        // Check for convergence
        int converged = 1;
        for (int i = 0; i < n - 1; i++) {
            if (fabs(H[i + 1][i]) > tol) {
                converged = 0;
                break;
            }
        }

        if (converged) {
            break;
        }
    }

    end = clock();
    double elapsedTime = ((double)(end - start)) / CLOCKS_PER_SEC;

    // Extract eigenvalues
    for (int i = 0; i < n; i++) {
        if (i < n - 1 && fabs(H[i + 1][i]) > tol) {
            // Complex eigenvalues from 2x2 submatrix
            double a = H[i][i];
            double b = H[i][i + 1];
            double c = H[i + 1][i];
            double d = H[i + 1][i + 1];
            double trace = a + d;
            double determinant = a * d - b * c;
            double discriminant = trace * trace - 4 * determinant;

            if (discriminant < 0) {
                double realPart = trace / 2.0;
                double imagPart = sqrt(-discriminant) / 2.0;
                eigenvalues[i] = realPart + imagPart * I;
                eigenvalues[i + 1] = realPart - imagPart * I;
            }
            i++; // Skip the next index as it's part of this 2x2 block
        } else {
            // Real eigenvalue
            eigenvalues[i] = H[i][i];
        }
    }

    printf("Number of iterations: %d\n", iter + 1);
    printf("Time required: %.6f seconds\n", elapsedTime);

    return iter + 1;
}

int main() {
    int n, maxIter = 1000;
    double tol = 1e-8;

    printf("Enter the size of the matrix (n x n): ");
    scanf("%d", &n);

    double **A = allocateMatrix(n);
    printf("Enter the elements of the matrix row by row:\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            scanf("%lf", &A[i][j]);
        }
    }

    hessenbergReduction(A, n);

    double complex *eigenvalues = (double complex *)malloc(n * sizeof(double complex));
    qrIteration(A, n, maxIter, tol, eigenvalues);

    printf("Eigenvalues:\n");
    for (int i = 0; i < n; i++) {
        printf("%10.6f + %10.6fi\n", creal(eigenvalues[i]), cimag(eigenvalues[i]));
    }

    freeMatrix(A, n);
    free(eigenvalues);
    return 0;
}

