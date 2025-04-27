
import numpy as np
import matplotlib.pyplot as plt
import math as m



def read_matrix_from_file(filename):
    """
    Считывает матрицу double-ов из файла.
    Каждая строка файла соответствует строке матрицы.
    Числа в строках файла должны быть разделены пробелами.
    
    Args:
        filename (str): Имя файла для чтения
        
    Returns:
        list[list[float]]: Считанная матрица
    """
    matrix = []
    with open(filename, 'r') as file:
        file.readline()
        for line in file:
            # Удаляем начальные/конечные пробелы и разбиваем строку по пробелам
            row = [float(num) for num in line.strip().split()]
            matrix.append(row)
    return np.array(matrix)

# Пример использования

filename = "results/results_0.txt"  # Замените на имя вашего файла


with open(filename, 'r') as file:
    rank, commsize = file.readline().strip().split()


matrix = read_matrix_from_file(f"results/results_{0}.txt")
matrix = matrix[:,:-1]
result_matrix = matrix
print(matrix.shape)

print(rank, commsize)
for m in range(1, int(commsize)-1):
    matrix = read_matrix_from_file(f"results/results_{m}.txt") 
    matrix = matrix[:,1:-1]
    result_matrix = np.hstack((result_matrix, matrix))

matrix = read_matrix_from_file(f"results/results_{int(commsize) - 1}.txt")
matrix = matrix[:,1:]
result_matrix = np.hstack((result_matrix, matrix))

N_tau, M_h = result_matrix.shape   

print(N_tau, M_h) 


plt.figure()
x = np.linspace(0, 1, M_h)
for t in [int(x) for x in np.linspace(0, N_tau-1, 10)]:
    plt.plot(x, result_matrix[t])
plt.show()