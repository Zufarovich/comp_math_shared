import matplotlib.pyplot as plt
import numpy as np


def weighted_scheme(a: int, f, phi, 
                    l_bound, r_bound,
                    sigma = 1/2, 
                    x_border = 1, t_border = 1,
                    x_num = int(1e3), t_num = int(1e3)) -> np.ndarray:
    
    t_grid = np.linspace(0, t_border, t_num)
    l_bound_gr = np.array([l_bound(t) for t in t_grid])
    r_bound_gr = np.array([r_bound(t) for t in t_grid])
    h = x_border / (x_num - 1)
    tau = t_border / t_num
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
        
        return np.linalg.solve(A, b)
    
    solution_grid[0] = phi(np.linspace(0, x_border, x_num))
    for t_layer in range(1, t_num):
        solution_grid[t_layer] = next_layer(solution_grid[t_layer - 1],
                                            t_layer)

    return(solution_grid)

a = 1
phi = lambda x: np.zeros_like(x)  # начальная температура: 0
l_bound = lambda t: 0
r_bound = lambda t: 0
f = lambda x, t: np.sin(np.pi * x)  # теплоисточник

# === Оценка порядка аппроксимации ===
C = 0.5  # tau = C * h
hs = [1 / n for n in [10, 20, 40, 80, 160]]
errors = []

u_exact = lambda x, t: np.sin(np.pi * x) * np.exp(-np.pi**2 * t)

for h in hs:
    x_num = int(1 / h) + 1
    tau = C * h
    t_num = int(1 / tau)

    sol = weighted_scheme(
        a=a,
        f=f,
        phi=phi,
        l_bound=l_bound,
        r_bound=r_bound,
        sigma=0.5,
        x_border=1,
        t_border=1,
        x_num=x_num,
        t_num=t_num
    )

    x_vals = np.linspace(0, 1, x_num)
    exact = u_exact(x_vals, 1.0)
    error = np.max(np.abs(sol[-1] - exact))
    
    # Проверяем, что ошибка не слишком мала
    if error > 1e-15:
        errors.append(error)
    else:
        errors.append(1e-15)  # устанавливаем минимальную ошибку

# Построим логарифмический график ошибки
hs_log = np.log10(hs)
errors_log = np.log10(errors)

# Проверяем, что ошибка логарифмична
if len(errors_log) > 1:
    slope, _ = np.polyfit(hs_log, errors_log, 1)
    print(f"Порядок аппроксимации (наклон log(error) vs log(h)): {abs(slope):.2f}")
else:
    print("Ошибка слишком мала для корректного вычисления порядка.")

# Построение графика
plt.figure()
plt.plot(hs_log, errors_log, 'o-', label='Ошибка аппроксимации')
plt.xlabel('log10(h)')
plt.ylabel('log10(ошибка)')
plt.title('График log(error) от log(h)')
plt.grid(True)
plt.legend()
plt.show()