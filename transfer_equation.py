import sys
import math
import numpy as np
import matplotlib.pyplot as plt
# Параметры задачи
N_x = 1000
M_tau = 10000

def phi_func(n: int) -> float:
    x = n / N_x
    if x < 0.5:
        sin_val = math.sin(2 * math.pi * x)
        return sin_val * sin_val
    else:
        return 0.0

def four_points_next_layer(previous: list[float],
                           next_row:    list[float],
                           a: float, f: list[list[float]], tau: float, h: float) -> None:
    coef = a * tau / (2.0 * h)
    # внутренние точки
    for i in range(1, N_x - 1):
        next_row[i] = (previous[i]
                       - coef * (previous[i+1] - previous[i-1])
                       + tau * f[0][i])
    # правая граница
    next_row[N_x - 1] = (previous[N_x - 1]
                         - 2*coef * (previous[N_x - 1] - previous[N_x - 2])
                         + tau * f[0][N_x - 1])

def cross_next_layer(grid: list[list[float]],
                     j:    int,
                     a:    float, f: list[list[float]], tau: float, h: float) -> None:
    coef = a * tau / h
    # внутренние точки
    for i in range(1, N_x - 1):
        grid[j][i] = (grid[j-2][i]
                      + 2*tau * f[j-1][i]
                      - coef * (grid[j-1][i+1] - grid[j-1][i-1]))
    # правая граница
    grid[j][N_x-1] = (grid[j-1][N_x-1]
                      - coef * (grid[j-1][N_x-1] - grid[j-1][N_x-2])
                      + tau * f[j-1][N_x-1])

def main():
    if len(sys.argv) < 2:
        print("You have to enter coefficient a too!")
        sys.exit(1)

    a = float(sys.argv[1])
    tau = 1.0 / M_tau
    h   = 1.0 / N_x

    # Проверка числа Куррана
    if a * tau / h > 1.0:
        print(f"The Courant Number {a*tau/h:g} > 1!")
        sys.exit(1)

    # Источники f[j][i] (M_tau×N_x), инициализируем нулями
    f = [[0.0]*N_x for _ in range(M_tau)]

    # Основной сеточный массив решений (M_tau×N_x)
    grid_solution = [[0.0]*N_x for _ in range(M_tau)]

    # Начальное условие в пространственной переменной (t=0)
    for n in range(N_x):
        grid_solution[0][n] = phi_func(n)
    # Граничное условие слева x=0 для всех t
    for k in range(M_tau):
        grid_solution[k][0] = 0.0

    # Первый шаг
    four_points_next_layer(grid_solution[0],
                           grid_solution[1],
                           a, f, tau, h)

    # Все последующие слои
    for j in range(2, M_tau):
        cross_next_layer(grid_solution, j, a, f, tau, h)

    grid_solution  = np.array(grid_solution)

   

    plt.figure()
    x = np.linspace(0, 1, N_x)
    for t in [int(x) for x in np.linspace(0, M_tau-1, 10)]:
        plt.plot(x, grid_solution[t])
    plt.savefig("filename.png")

if __name__ == "__main__":
    main()
