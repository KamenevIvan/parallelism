import matplotlib.pyplot as plt
import numpy as np

# Читаем данные из файла
filename = "data2.txt"  # Замени на имя твоего файла
data = np.loadtxt(filename)

threads = data[:, 0]  # Кол-во потоков
execution_time = data[:, 1]  # Время выполнения
speedup = data[:, 2]  # Ускорение

# Создаем графики
plt.figure(figsize=(10, 5))

# График времени выполнения
plt.subplot(1, 2, 1)
plt.plot(threads, execution_time, marker='o', linestyle='-')
plt.xlabel("Number of Threads")
plt.ylabel("Execution Time")
plt.title("Execution Time vs Number of Threads")
plt.grid(True)

# График ускорения
plt.subplot(1, 2, 2)
plt.plot(threads, speedup, marker='s', linestyle='-', color='r')
plt.xlabel("Number of Threads")
plt.ylabel("Speedup")
plt.title("Speedup vs Number of Threads")
plt.grid(True)

plt.tight_layout()
plt.show()
