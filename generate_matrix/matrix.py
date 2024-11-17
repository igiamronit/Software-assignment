import random

def generate_random_square_matrix(n, lower_bound=-10, upper_bound=10):
    """
    Generates a random square matrix with dimensions n x n.
    
    Parameters:
        n (int): The number of rows and columns (matrix size).
        lower_bound (int): Minimum value of the matrix elements (default: -10).
        upper_bound (int): Maximum value of the matrix elements (default: 10).
        
    Returns:
        list[list[int]]: A square matrix of size n x n.
    """
    return [[random.randint(lower_bound, upper_bound) for _ in range(n)] for _ in range(n)]

# Example usage
n = int(input("Enter the size of the square matrix: "))
matrix = generate_random_square_matrix(n)

print("Generated Random Matrix:")
for row in matrix:
    print(" ".join(map(str, row)))

