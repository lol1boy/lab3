import numpy as np

x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
              13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24], dtype=float)
y = np.array([-2, 0, 5, 10, 15, 20, 23, 22, 17, 10, 5, 0,
              -10, 3, 7, 13, 19, 20, 22, 21, 18, 15, 10, 3], dtype=float)


def form_matrix(x, m):
    A = np.zeros((m+1, m+1), dtype=float)

    for i in range(m + 1):
        for j in range(m + 1):
            A[i, j] = np.sum(x ** (i + j))

    return A

def form_vector(x, y, m):
    b = np.zeros(m+1)
    for i in range(0, m):
        b[i] = np.sum(y * x ** i)
    
    return b

def gauss_solve(A_in, b_in):
    n = len(b_in)
    A = A_in.copy().astype(float)
    b = b_in.copy().astype(float)

    for k in range(n):
        max_row = k + np.argmax(np.abs(A[k:, k]))
        A[[k, max_row]] = A[[max_row, k]]
        b[[k, max_row]] = b[[max_row, k]]

        for i in range(k + 1, n):
            if A[k, k] == 0:
                raise ValueError("Singular matrix - no unique solution")
            factor = A[i, k] / A[k, k]
            A[i, k:] -= factor * A[k, k:]
            b[i] -= factor * b[k]

        x_sol = np.zeros(n, dtype=float)
        for i in range(n - 1, -1, -1):
            x_sol[i] = (b[i] - np.dot(A[i, i+1:], x_sol[i+1:])) / A[i, i]

        return x_sol

def polynomial(x, coef):
    return sum(coef[i] * x**i for i in range(len(coef)))

def compute_varince(y_true, y_approx):
    return np.mean((y_true - y_approx) ** 2)

variances = []
for m in range(1, 7):
    A = form_matrix(x, m)
    b = form_vector(x, y, m)
    coef = gauss_solve(A, b)
    y_approx = polynomial(x, coef)
    variances.append(compute_varince(y, y_approx))
    print(f"m={m} variance={variances[-1]:.4f}")

optimal_m = np.argmin(variances) + 1
print(f"\nOptimal degree: m = {optimal_m}")

A = form_matrix(x, optimal_m)
b = form_vector(x, y, optimal_m)
coef = gauss_solve(A, b)

y_approx = polynomial(x, coef)
forecast = polynomial(np.array([25.0, 26.0, 27.0]), coef)
print(f"Forecast months 25-27: {np.round(forecast, 2)}")

    
# print(form_matrix(x, 3))