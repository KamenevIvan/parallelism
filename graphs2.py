import matplotlib.pyplot as plt
import pandas as pd
import math

# Читаем данные, игнорируя строки, начинающиеся с #
filename = "data3.txt"  # Замени на имя твоего файла
df = pd.read_csv(filename, delim_whitespace=True, comment="#", header=None, names=["threads", "schedule", "chunk_size", "time", "speedup"])

# Определяем уникальные размеры чанков
chunk_sizes = sorted(df["chunk_size"].unique())  # Сортируем размеры чанков

# Определяем количество строк и колонок для графиков
num_chunks = len(chunk_sizes)
cols = 2  # Число столбцов (можно менять)
rows = math.ceil(num_chunks / cols)  # Определяем число строк

# Создаем графики
plt.figure(figsize=(12, 6 + 2 * rows))  # Динамический размер

for i, chunk in enumerate(chunk_sizes, 1):
    plt.subplot(rows, cols, i)
    
    # Фильтруем данные по текущему размеру чанка
    df_chunk = df[df["chunk_size"] == chunk]
    
    for schedule in df_chunk["schedule"].unique():
        df_schedule = df_chunk[df_chunk["schedule"] == schedule]
        plt.plot(df_schedule["threads"], df_schedule["speedup"], marker='o', linestyle='-', label=schedule)
    
    plt.xlabel("Number of Threads")
    plt.ylabel("Speedup")
    plt.title(f"Speedup vs Threads (Chunk Size = {chunk})")
    plt.legend()
    plt.grid(True)

plt.tight_layout()
plt.show()
