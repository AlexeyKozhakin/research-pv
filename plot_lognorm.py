import numpy as np
import matplotlib.pyplot as plt

# заданные параметры для логнормальной случайной величины
M = 100  # среднее
S = 50   # стандартное отклонение

# пересчёт параметров для логнормального распределения (параметры логарифма)
sigma_sq = np.log(1 + (S**2) / (M**2))
mu = np.log(M) - sigma_sq / 2
sigma = np.sqrt(sigma_sq)

# размер выборки
size = 1000

# генерация выборки
deposit_sizes = np.random.lognormal(mean=mu, sigma=sigma, size=size)

# проверим фактические среднее и стандартное отклонение
print(f'Выборочное среднее: {np.mean(deposit_sizes):.2f}')
print(f'Выборочное стандартное отклонение: {np.std(deposit_sizes):.2f}')

# гистограмма
plt.hist(deposit_sizes, bins=50, density=True, alpha=0.6, color='skyblue')
plt.title('Размеры депозитов (логнормальное распределение)')
plt.xlabel('Размер депозита')
plt.ylabel('Плотность')
plt.show()
