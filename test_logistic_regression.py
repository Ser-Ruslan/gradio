import numpy as np
from logistic_regression import LogisticRegressionModel

model = LogisticRegressionModel(b0=-10, B=np.array([0.2, 0.8]))

# Тест 1: Высокие показатели
X1 = np.array([60.0, 45.0])
p1 = model.predict(X1)
c1 = model.predict_class(X1)
print(f"Тест 1: {c1} (вероятность: {p1:.3f}) - {'Пройден' if c1 == 1 and p1 > 0.5 else 'Не пройден'}")

# Тест 2: Низкие показатели
X2 = np.array([15.0, 15.0])
p2 = model.predict(X2)
c2 = model.predict_class(X2)
print(f"Тест 2: {c2} (вероятность: {p2:.3f}) - {'Пройден' if c2 == 0 and p2 < 0.5 else 'Не пройден'}")

# Тест 3: Средние показатели
X3 = np.array([35.0, 30.0])
p3 = model.predict(X3)
c3 = model.predict_class(X3)
print(f"Тест 3: {c3} (вероятность: {p3:.3f}) - {'Пройден' if c3 == 0 and p3 < 0.5 else 'Не пройден'}")
