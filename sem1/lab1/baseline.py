import random
import numpy as np
import matplotlib.pyplot as plt

# ---//--- Generate Data
dec = 1000
coef = [1, 5e-3, 1e-4]

x = np.array([i - 1 for i in range(dec)])
y = np.array([coef[0] + coef[1] * (i - 1) + coef[2] * (i - 1) ** 2 for i in range(dec)])

y_noise = y + np.random.normal(0, 5, 1000)

# ---//--- Least squares method
# H: y_hat = w_0* + w_1* * x + w_2* * x^2
# Metrics: RMS error
# Q( w_0*, w_1*, w_2*, x ) = 1/n * sum( ( y_hat - y_i )**2 )

# For polynom: X^T * X * a = X^T * y,
# where X - Vandermond matrix ( X = [ 1, x_1, ... x_1^m] ), a = [ a_0, a_1, ... a_m ]^T
# Let F = X^T * X ==> inv(F) = 1/det( F ) * (minor(F))^T
# Let K = X^T * y
# then a = inv(F) * K


def LSM_coef_calc(x, y, degree):

    n = len(x)
    m = degree + 1

    #  Matrix A (left side)
    F = [[0] * m for _ in range(m)]
    K = []
    for i in range(m):
        for j in range(m):
            F[i][j] = sum(x_i ** (i + j) for x_i in x)

        K.append(sum(y_i * (x_i**i) for x_i, y_i in zip(x, y)))

    coef = np.linalg.solve(F, K)

    # Coefficients:
    return coef


LSM_coef = LSM_coef_calc(x, y_noise, 2)
print(f"Уравнение: y = {LSM_coef[2]:.6f}*x^2 + {LSM_coef[1]:.6f}*x + {LSM_coef[0]:.6f}")

y_hat = [
    np.dot(LSM_coef, [x[i] ** j for j in range(0, len(LSM_coef))])
    for i in range(0, len(x))
]

# Plot
plt.plot(x, y, color="k")
plt.plot(x, y_noise, "o", markersize=0.5)
plt.plot(x, y_hat, color="r")
plt.xlabel("x", fontsize=20)
plt.ylabel("y", fontsize=20)
plt.grid(True)
plt.legend({"Истинный сигнал", "Зашумленные данные", "МНК"})
plt.show()
