import numpy as np
import matplotlib.pyplot as plt
from   scipy.optimize import curve_fit
from   tqdm import tqdm 


def u(x, t, num_terms=100):
    x = np.atleast_1d(x)
    t = np.atleast_1d(t)
    x, t = np.meshgrid(x, t)
    total = np.zeros_like(x)
    for n in range(1, num_terms + 1):
        coef = 8 * (1 - (-1)**n) / (np.pi * n**3)
        total += coef * np.exp(-n**2 * t) * np.sin(n * x)
    return total

##################################################
##################################################

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
        
        return tma(A, b)
    
    solution_grid[0] = phi(np.linspace(0, x_border, x_num))
    for t_layer in range(1, t_num):
        solution_grid[t_layer] = next_layer(solution_grid[t_layer - 1],
                                            t_layer)
    

    return(solution_grid)


def tma(A, b):
    n = A.shape[0]
    x = np.zeros(n, dtype=np.double)

    alpha = np.zeros(n, dtype=np.double)
    betta = np.zeros(n, dtype=np.double)

    alpha[1] = - A[0][1]/A[0][0]
    betta[1] = b[0]/A[0][0]

    for i in range(2, n):
        denominator = 1 / (A[i - 1][i - 2] * alpha[i - 1] + A[i - 1][i - 1])
        alpha[i] = - A[i - 1][i] * denominator
        betta[i] = (b[i - 1] - A[i - 1][i - 2] * betta[i - 1]) * denominator

    x[-1] = (b[-1] - A[-1][-2] * betta[-1])/(A[-1][-2] * alpha[-1] + A[-1][-1])

    for i in range(A.shape[0] - 2, -1, -1):
        x[i] = alpha[i + 1] * x[i + 1] + betta[i + 1] 

    return x


def f_zer(a, b):
    return 0

def phi(x):
    return x * (np.pi - x)

def l_bound(t):
    return  0

def r_bound(t):
    return 0


# Параметры графика
#t_fixed = 0.1  # Фиксированное значение t
t = np.linspace(0, 1, 20)
u_x_t = np.zeros_like(t)

my_solution = weighted_scheme(1, f_zer, phi, 
                l_bound, r_bound,
                sigma = 1/2, 
                x_border = np.pi , t_border = 1,
                x_num = int(1e2), t_num = int(1e3)) 


# Построение графика
# Параметры графика

t_border = 1
x_border = np.pi
sigma = 0.5
x_num = int(1e3)
t_num = int(1e3)

my_solution = weighted_scheme(1, f_zer, phi, 
                l_bound, r_bound,
                sigma = sigma, 
                x_border = x_border , t_border = t_border,
                x_num = x_num, t_num = t_num) 


x_values = np.linspace(0, x_border, x_num) 
t_values = np.linspace(0, t_border, t_num)
t_grid_ind = [10, 200, 800]

# Построение графика
plt.figure(figsize=(10, 6))

for ind in t_grid_ind:

    u_values = [u(x, t_values[ind], num_terms=50) for x in x_values] 

    solution_for_graph = my_solution[ind, :]

    plt.plot(x_values, u_values, label=f'u_true(x, t={t_values[ind]})', linestyle='dashed')
    plt.plot(x_values, solution_for_graph, label=f'u_our(x, t={t_values[ind]})')

plt.title('Графики функции u(x, t) при фиксированном t')
plt.xlabel('x')
plt.ylabel('u(x, t)')
plt.grid(True)
plt.legend()
plt.show()


loss  = []
steps = np.linspace(1e2, 500, 100, dtype=int)


for n_step in tqdm(steps):
    solution = weighted_scheme(1, f_zer, phi, 
                l_bound, r_bound,
                sigma = 1/2, 
                x_border = np.pi , t_border = 1,
                x_num = n_step, t_num = n_step) 
    
    t_grid = np.linspace(0, 1, n_step)
    x_grid = np.linspace(0, np.pi, n_step)

    true_value = np.zeros_like(solution)
    for t in t_grid:
        true_value[int(t*n_step-1)] =u(x_grid, t, num_terms=50)


    loss.append(np.max(np.abs(solution - true_value)))

def linear_regr(x, a, b):
    return (a*x + b)

log_h = np.log(np.pi / (steps - 1))
log_loss = np.log(loss)
popt, pcov = curve_fit(linear_regr, log_h, log_loss)
print("Порядок сходимости из наклона прямой:", popt[0])


plt.scatter(log_h, log_loss)
plt.plot(log_h, linear_regr(log_h, popt[0], popt[1]), color='red')
plt.grid()
plt.show()