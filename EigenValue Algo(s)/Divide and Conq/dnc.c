#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define MAX_SIZE 100
#define EPSILON 1e-10
#define MAX_ITER 100

// Function to create a matrix
double** createMatrix(int n) {
    double** matrix = (double**)malloc(n * sizeof(double*));
    for (int i = 0; i < n; i++) {
        matrix[i] = (double*)malloc(n * sizeof(double));
    }
    return matrix;
}

// Function to free matrix memory
void freeMatrix(double** matrix, int n) {
    for (int i = 0; i < n; i++) {
        free(matrix[i]);
    }
    free(matrix);
}

// Function to copy matrix
void copyMatrix(double** src, double** dest, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            dest[i][j] = src[i][j];
        }
    }
}

// Function to check if matrix is diagonal within epsilon
int isDiagonal(double** matrix, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i != j && fabs(matrix[i][j]) > EPSILON) {
                return 0;
            }
        }
    }
    return 1;
}

// Function to perform Householder transformation to tridiagonal form
void householderToTridiagonal(double** matrix, int n) {
    for (int k = 0; k < n - 2; k++) {
        double alpha = 0.0;
        for (int i = k + 1; i < n; i++) {
            alpha += matrix[i][k] * matrix[i][k];
        }
        
        if (alpha < EPSILON) continue;
        
        alpha = sqrt(alpha);
        if (matrix[k + 1][k] > 0) alpha = -alpha;
        
        double r = sqrt(0.5 * (alpha * alpha - alpha * matrix[k + 1][k]));
        double* v = (double*)calloc(n, sizeof(double));
        
        v[k + 1] = (matrix[k + 1][k] - alpha) / (2 * r);
        for (int i = k + 2; i < n; i++) {
            v[i] = matrix[i][k] / (2 * r);
        }
        
        // Apply Householder transformation
        double** temp = createMatrix(n);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0.0;
                for (int l = k + 1; l < n; l++) {
                    sum += v[l] * matrix[l][j];
                }
                temp[i][j] = matrix[i][j] - 2 * v[i] * sum;
            }
        }
        
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double sum = 0.0;
                for (int l = k + 1; l < n; l++) {
                    sum += temp[i][l] * v[l];
                }
                matrix[i][j] = temp[i][j] - 2 * sum * v[j];
            }
        }
        
        freeMatrix(temp, n);
        free(v);
    }
}

// Function to perform QR iteration
void qrIteration(double** matrix, int n) {
    double** Q = createMatrix(n);
    double** R = createMatrix(n);
    
    for (int iter = 0; iter < MAX_ITER; iter++) {
        // Compute Q and R matrices using Givens rotations
        copyMatrix(matrix, R, n);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                Q[i][j] = (i == j) ? 1.0 : 0.0;
            }
        }
        
        for (int i = 0; i < n - 1; i++) {
            for (int j = i + 1; j < n; j++) {
                if (fabs(R[j][i]) > EPSILON) {
                    double r = sqrt(R[i][i] * R[i][i] + R[j][i] * R[j][i]);
                    double c = R[i][i] / r;
                    double s = -R[j][i] / r;
                    
                    // Update R
                    for (int k = i; k < n; k++) {
                        double temp = R[i][k];
                        R[i][k] = c * temp - s * R[j][k];
                        R[j][k] = s * temp + c * R[j][k];
                    }
                    
                    // Update Q
                    for (int k = 0; k < n; k++) {
                        double temp = Q[k][i];
                        Q[k][i] = c * temp - s * Q[k][j];
                        Q[k][j] = s * temp + c * Q[k][j];
                    }
                }
            }
        }
        
        // Compute new matrix = RQ
        double** temp = createMatrix(n);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                temp[i][j] = 0.0;
                for (int k = 0; k < n; k++) {
                    temp[i][j] += R[i][k] * Q[k][j];
                }
            }
        }
        
        copyMatrix(temp, matrix, n);
        freeMatrix(temp, n);
        
        if (isDiagonal(matrix, n)) break;
    }
    
    freeMatrix(Q, n);
    freeMatrix(R, n);
}

// Main function to find eigenvalues using divide and conquer
void findEigenvalues(double** matrix, int n) {
    // First reduce to tridiagonal form
    householderToTridiagonal(matrix, n);
    
    // Then use QR iteration to find eigenvalues
    qrIteration(matrix, n);
    
    // Eigenvalues are now on the diagonal
    printf("Eigenvalues:\n");
    for (int i = 0; i < n; i++) {
        printf("%.6f\n", matrix[i][i]);
    }
}

int main() {
    int n;
    printf("Enter the size of the symmetric matrix: ");
    scanf("%d", &n);
    
    if (n > MAX_SIZE) {
        printf("Matrix size too large. Maximum allowed size is %d\n", MAX_SIZE);
        return 1;
    }
    
    double** matrix = createMatrix(n);
    
    printf("Enter the matrix elements row by row:\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            scanf("%lf", &matrix[i][j]);
        }
    }
    
    // Verify symmetry
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            if (fabs(matrix[i][j] - matrix[j][i]) > EPSILON) {
                printf("Error: Matrix is not symmetric!\n");
                freeMatrix(matrix, n);
                return 1;
            }
        }
    }
    
    findEigenvalues(matrix, n);
    
    freeMatrix(matrix, n);
    return 0;
}
