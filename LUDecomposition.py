import numpy as np

def lu_decomposition(A):
    """
    Performs LU decomposition of a square matrix A.

    Args:
        A: The square matrix (numpy array).

    Returns:
        A tuple (L, U) where L is the lower triangular matrix and U is the upper triangular matrix.
        Returns None if A is singular (cannot be decomposed).
    """

    n = A.shape[0]
    L = np.eye(n)  # Initialize L as an identity matrix
    U = A.copy()   # Initialize U as a copy of A

    for i in range(n):
        if U[i, i] == 0:
            return None  # Singular matrix, cannot decompose

        for j in range(i + 1, n):
            factor = U[j, i] / U[i, i]
            L[j, i] = factor
            U[j, i:] = U[j, i:] - factor * U[i, i:]

    return L, U

def solve_lu(L, U, b):
    """
    Solves Ax = b using LU decomposition (Ly = b, Ux = y).

    Args:
        L: The lower triangular matrix from LU decomposition.
        U: The upper triangular matrix from LU decomposition.
        b: The right-hand side vector (numpy array).

    Returns:
        The solution vector x (numpy array).
    """

    n = len(b)
    y = np.zeros(n)
    x = np.zeros(n)

    # Forward substitution (Ly = b)
    y[0] = b[0] / L[0, 0]
    for i in range(1, n):
        y[i] = (b[i] - np.dot(L[i, :i], y[:i])) / L[i, i]

    # Backward substitution (Ux = y)
    x[n - 1] = y[n - 1] / U[n - 1, n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i + 1:], x[i + 1:])) / U[i, i]

    return x

# Example usage:
A = np.array([[2, 1, -1],
              [-3, -1, 2],
              [-2, 1, 2]], dtype=float)
b = np.array([8, -11, -3], dtype=float)

L, U = lu_decomposition(A)

if L is not None and U is not None:
    print("L:")
    print(L)
    print("U:")
    print(U)

    x = solve_lu(L, U, b)
    print("Solution:")
    print(x)
else:
    print("Matrix is singular, LU decomposition failed.")

# Example with a singular matrix:
A_singular = np.array([[1, 2], [2, 4]], dtype=float)
b_singular = np.array([1, 2], dtype=float)

L_singular, U_singular = lu_decomposition(A_singular)

if L_singular is not None and U_singular is not None:
    print("L:")
    print(L_singular)
    print("U:")
    print(U_singular)

    x_singular = solve_lu(L_singular, U_singular, b_singular)
    print("Solution:")
    print(x_singular)
else:
    print("Matrix is singular, LU decomposition failed.")