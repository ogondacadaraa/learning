import numpy as np

def gauss_elimination(A, b):
    """
    Solves a system of linear equations Ax = b using Gaussian elimination.

    Args:
        A: The coefficient matrix (numpy array).
        b: The right-hand side vector (numpy array).

    Returns:
        The solution vector x (numpy array) if a unique solution exists,
        or None if the system is singular or inconsistent.
    """

    n = len(b)
    Ab = np.concatenate((A, b.reshape(n, 1)), axis=1)  # Augmented matrix

    # Forward elimination
    for i in range(n):
        # Partial pivoting (optional but improves numerical stability)
        max_row = i
        for k in range(i + 1, n):
            if abs(Ab[k, i]) > abs(Ab[max_row, i]):
                max_row = k
        Ab[[i, max_row]] = Ab[[max_row, i]]

        if Ab[i, i] == 0:
            return None  # Singular matrix

        for j in range(i + 1, n):
            factor = Ab[j, i] / Ab[i, i]
            Ab[j, i:n + 1] = Ab[j, i:n + 1] - factor * Ab[i, i:n + 1]

    # Back substitution
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (Ab[i, n] - np.dot(Ab[i, i + 1:n], x[i + 1:n])) / Ab[i, i]

    return x

# Example usage:
A = np.array([[10, 1, -1],
              [-3, -1, 2],
              [-2, 1, 11]], dtype=float)
b = np.array([8, -11, -3], dtype=float)

x = gauss_elimination(A, b)

if x is not None:
    print("Solution:")
    print(x)
else:
    print("The system is singular or inconsistent.")

# Example with a singular matrix:
A_singular = np.array([[1, 2], [2, 4]], dtype=float)
b_singular = np.array([1, 2], dtype=float)

x_singular = gauss_elimination(A_singular, b_singular)

if x_singular is not None:
    print("Solution:")
    print(x_singular)
else:
    print("The system is singular or inconsistent.")