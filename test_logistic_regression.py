import numpy as np
from logistic_regression import LinearRegressionModel

model = LinearRegressionModel(b0=-10, B=np.array([0.2, 0.8]))

# Тест 1: Высокие показатели (ожидаем положительное значение)
X1 = np.array([60.0, 45.0])
v1 = model.predict(X1)
print(f"Тест 1 (высокие показатели): значение = {v1:.3f} - {'Пройден' if v1 > 0 else 'Не пройден'}")     

# Тест 2: Низкие показатели (ожидаем отрицательное значение)
X2 = np.array([15.0, 15.0])
v2 = model.predict(X2)
print(f"Тест 2 (низкие показатели): значение = {v2:.3f} - {'Пройден' if v2 < 0 else 'Не пройден'}")

# Тест 3: Средние показатели (ожидаем значение близкое к 0)
X3 = np.array([35.0, 30.0])
v3 = model.predict(X3)
print(f"Тест 3 (средние показатели): значение = {v3:.3f} - {'Пройден' if abs(v3) < 5 else 'Не пройден'}")

