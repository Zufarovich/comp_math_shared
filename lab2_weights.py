import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from tqdm import tqdm
from scipy.interpolate import interp1d

def u(x, t, num_terms=100):
    total = 0.0
    for n in range(1, num_terms + 1):
        coefficient = 8 * (1 - (-1)**n) / (np.pi * n**3)
        exponent = -n**2 * t
        total += coefficient * np.exp(exponent) * np.sin(n * x)
    return total/2

def weighted_scheme(a: int, f, phi,
                    l_bound, r_bound,
                    sigma = 1/2,
                    x_border = 1, t_border = 1,
                    x_num = int(1e3), t_num = int(1e3)) -> np.ndarray:

    t_grid = np.linspace(0, t_border, t_num)
    x_grid = np.linspace(0, x_border, x_num)
    l_bound_gr = np.array([l_bound(t) for t in t_grid])
    r_bound_gr = np.array([r_bound(t) for t in t_grid])
    h = x_border / (x_num - 1)
    tau = t_border / (t_num - 1)
    a_2 = a**2
    h_2 = h**2

    solution_grid = np.zeros((t_num, x_num))

    def next_layer(y_j:np.ndarray, t_layer) -> np.ndarray:
        A = np.zeros((x_num, x_num))
        b = np.zeros(x_num)

        A[0][0] = 1
        A[-1][-1] = 1
        b[0] =  l_bound_gr[t_layer]
        b[-1] = r_bound_gr[t_layer]
        for index in range(1, x_num - 1):
            A[index][index - 1] = -sigma*a_2/h_2
            A[index][index] = (1 / tau + 2*sigma*a_2/h_2)
            A[index][index + 1] = -sigma*a_2/h_2

            b[index] = (
                        y_j[index]/tau
                        + (1 - sigma)*a_2/h_2*(y_j[index + 1] - 2*y_j[index] + y_j[index - 1])
                        + f(h*index, tau*t_layer + sigma*tau)
                    )

        return tma(A, b) #Использовать исправленную версию tma

    solution_grid[0] = phi(x_grid)
    for t_layer in range(1, t_num):
        solution_grid[t_layer] = next_layer(solution_grid[t_layer - 1],
                                            t_layer)


    return solution_grid, x_grid, t_grid

def tma(A, b): # исправленная версия
    n = A.shape[0]
    x = np.zeros(n, dtype=np.double)

    alpha = np.zeros(n, dtype=np.double)
    beta = np.zeros(n, dtype=np.double)

    # Прямой ход
    alpha[0] = -A[0, 1] / A[0, 0]  # Коррекция индекса
    beta[0] = b[0] / A[0, 0]   # Коррекция индекса

    for i in range(1, n):
        denominator = A[i, i] + A[i, i - 1] * alpha[i - 1]
        alpha[i] = -A[i, i + 1] / denominator if i < n - 1 else 0 #Условие для последнего элемента
        beta[i] = (b[i] - A[i, i - 1] * beta[i - 1]) / denominator

    # Обратный ход
    x[n - 1] = beta[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = alpha[i] * x[i + 1] + beta[i]

    return x


def f_zer(x,t): #Добавлены аргументы x и t
    return 0

def phi(x):
    return x * (np.pi - x)

def l_bound(t):
    return  0

def r_bound(t):
    return 0


# Параметры графика
t_border = 1
x_border = np.pi
sigma = 0.5
x_num = int(1e3)
t_num = int(1e3)

my_solution, x_values, t_values = weighted_scheme(1, f_zer, phi,
                l_bound, r_bound,
                sigma = sigma,
                x_border = x_border , t_border = t_border,
                x_num = x_num, t_num = t_num)


t_grid_ind = [10, 200, 800]

# Построение графика
plt.figure(figsize=(10, 6))

for ind in t_grid_ind:

    u_values = [u(x, t_values[ind], num_terms=50) for x in x_values]

    solution_for_graph = my_solution[ind, :]

    plt.plot(x_values, u_values, label=f'u_true(x, t={t_values[ind]:.2f})', linestyle='dashed')
    plt.plot(x_values, solution_for_graph, label=f'u_our(x, t={t_values[ind]:.2f})')

plt.title('Графики функции u(x, t) при фиксированном t')
plt.xlabel('x')
plt.ylabel('u(x, t)')
plt.grid(True)
plt.legend()
plt.show()


loss  = []
steps = np.linspace(20, 500, 20, dtype=int) #Меньше точек, чтобы быстрее

for n_step in tqdm(steps):
    solution, x_grid, t_grid = weighted_scheme(1, f_zer, phi,
                l_bound, r_bound,
                sigma = 1/2,
                x_border = np.pi , t_border = 1,
                x_num = n_step, t_num = n_step)

    true_value = np.zeros_like(solution)
    for i, t in enumerate(t_grid):
        for j, x in enumerate(x_grid):
            true_value[i, j] = u(x, t, num_terms=50)


    loss.append(np.mean(np.abs(solution - true_value))) #Усреднение ошибки

def linear_regr(x, a, b):
    return (a*x + b)

log_h = np.log(1 / steps)
log_loss = np.log(loss)
popt, pcov = curve_fit(linear_regr, log_h, log_loss)
print("Порядок сходимости из наклона прямой:", popt[0])


plt.scatter(log_h, log_loss)
x_plot = np.linspace(min(log_h), max(log_h), 100) #Больше точек для линии
plt.plot(x_plot, linear_regr(x_plot, popt[0], popt[1]), color='red')
plt.xlabel("log(h)")
plt.ylabel("log(loss)")
plt.title("Оценка порядка сходимости")
plt.grid()
plt.show()