import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def solve_poisson_gauss_seidel(l, N, f, phi1, phi2, phi3, phi4):
    hx = l / (N - 1)
    hy = l / (N - 1)
    h_sq = hx**2

    u = np.zeros((N, N))
    x = np.linspace(0, l, N)
    y = np.linspace(0, l, N)
    X, Y = np.meshgrid(x, y, indexing='ij')

    u[:, 0] = phi1(x)  # Нижняя граница (y=0)
    u[:, -1] = phi2(x) # Верхняя граница (y=l)
    u[0, :] = phi3(y)  # Левая граница (x=0)
    u[-1, :] = phi4(y) # Правая граница (x=l)

    # Согласование углов (среднее арифметическое)
    u[0, 0] = 0.5 * (phi1(x[0]) + phi3(y[0]))
    u[-1, 0] = 0.5 * (phi1(x[-1]) + phi4(y[0]))
    u[0, -1] = 0.5 * (phi2(x[0]) + phi3(y[-1]))
    u[-1, -1] = 0.5 * (phi2(x[-1]) + phi4(y[-1]))

    F = f(X, Y) # Вычисляем правую часть на сетке

    tol = 1e-6       # Допуск для критерия остановки
    max_iter = 20000 # Максимальное количество итераций
    iterations = 0   # Счетчик итераций

    for k in range(max_iter):
        u_old = np.copy(u)

        # Итерация метода Зейделя
        for i in range(1, N-1):
            for j in range(1, N-1):
                u[i, j] = 0.25 * (u[i+1, j] + u[i-1, j] + \
                                  u[i, j+1] + u[i, j-1] + h_sq * F[i, j])

        norm_diff = np.linalg.norm(u - u_old, ord=np.inf)
        if norm_diff < tol:
            iterations = k + 1
            print(f"Метод Зейделя сошелся за {iterations} итераций.")
            break
    else:
        iterations = max_iter
        print(f"Метод Зейделя: Достигнуто максимальное количество итераций ({max_iter}). Норма разности: {norm_diff:.2e}")

    return X, Y, u, F, iterations

# --- Пример с известным точным решением ---

# Параметры
l = 1.0
N = 51 # N узлов -> N-1 интервалов -> h=l/(N-1) = 1/50

# Точное решение (для проверки)
u_exact_func = lambda x, y: np.sin(2 * np.pi * x) * np.cos(np.pi * y)

f = lambda x, y: 5 * np.pi**2 * np.sin(2 * np.pi * x) * np.cos(np.pi * y)

# Граничные условия, соответствующие u_exact
phi1 = lambda x: u_exact_func(x, 0)   # y=0 -> cos(0)=1 -> sin(2πx)
phi2 = lambda x: u_exact_func(x, l)   # y=l=1 -> cos(π)=-1 -> -sin(2πx)
phi3 = lambda y: u_exact_func(0, y)   # x=0 -> sin(0)=0 -> 0
phi4 = lambda y: u_exact_func(l, y)   # x=l=1 -> sin(2π)=0 -> 0

X, Y, U_numerical, F_grid, iters_gs = solve_poisson_gauss_seidel(l, N, f, phi1, phi2, phi3, phi4)
U_exact = u_exact_func(X, Y)

Error = np.abs(U_numerical - U_exact)
max_error = np.max(Error)
print(f"Максимальная абсолютная ошибка (Зейдель): {max_error:.2e}")

plt.figure(figsize=(18, 5))

# 1. Численное решение
plt.subplot(1, 3, 1)
contour_num = plt.contourf(X, Y, U_numerical, levels=50, cmap="viridis")
plt.colorbar(contour_num, label='u(x, y)')
plt.title(f"Численное решение (Зейдель, N={N}, iters={iters_gs})") # Добавлено кол-во итераций
plt.xlabel("x")
plt.ylabel("y")
plt.axis('equal')
plt.grid(True, linestyle=':', alpha=0.5)

# 2. Точное решение
plt.subplot(1, 3, 2)
contour_exact = plt.contourf(X, Y, U_exact, levels=50, cmap="viridis")
plt.colorbar(contour_exact, label='u_exact(x, y)')
plt.title("Точное решение")
plt.xlabel("x")
plt.ylabel("y")
plt.axis('equal')
plt.grid(True, linestyle=':', alpha=0.5)

# 3. Абсолютная ошибка
plt.subplot(1, 3, 3)
contour_err = plt.contourf(X, Y, Error, levels=50, cmap="Reds") # Другая карта для ошибки
plt.colorbar(contour_err, label='|u - u_exact|')
plt.title(f"Абсолютная ошибка (Max: {max_error:.2e})")
plt.xlabel("x")
plt.ylabel("y")
plt.axis('equal')
plt.grid(True, linestyle=':', alpha=0.5)

plt.tight_layout()
plt.show()

fig = plt.figure(figsize=(12, 5))

# Численное решение 3D
ax1 = fig.add_subplot(1, 2, 1, projection='3d')
surf1 = ax1.plot_surface(X, Y, U_numerical, cmap='viridis', edgecolor='none', rstride=2, cstride=2)
ax1.set_title("Численное решение U (Зейдель)")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_zlabel("U(x, y)")

# Точное решение 3D
ax2 = fig.add_subplot(1, 2, 2, projection='3d')
surf2 = ax2.plot_surface(X, Y, U_exact, cmap='viridis', edgecolor='none', rstride=2, cstride=2)
ax2.set_title("Точное решение U_exact")
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.set_zlabel("U_exact(x, y)")

plt.tight_layout()
plt.show()