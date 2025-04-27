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
        for line in file:
            # Удаляем начальные/конечные пробелы и разбиваем строку по пробелам
            row = [float(num) for num in line.strip().split()]
            matrix.append(row)
    return matrix

# Пример использования

filename = "results.txt"  # Замените на имя вашего файла
matrix = read_matrix_from_file(filename)
    

plt.figure()
x = np.linspace(0, 1, 1000)
for t in range (0, 1000, 100):
    plt.plot(x, matrix[t])
plt.savefig("test.png")