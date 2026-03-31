import numpy as np
import matplotlib.pyplot as plt
from typing import Union


class LinearRegressionModel:
    """
    Класс для имитации линейной регрессии с предопределенными коэффициентами.
    """
    
    def __init__(self, b0: float = -10, B: np.ndarray = None):
        """
        Инициализация модели с коэффициентами регрессии.
        
        Args:
            b0: свободный член 
            B: numpy массив весов признаков
        """
        self.b0 = b0  # свободный член
        if B is None:
            self.B = np.array([0.2, 0.8])  
        else:
            self.B = np.array(B)
    
    def predict(self, X: np.ndarray) -> float:
        """
        Применяет модель линейной регрессии к входному массиву.
        
        Args:
            X: numpy массив с входными данными
            
        Returns:
            float: предсказанное значение
        """
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        
        # Нормализация второго признака (points_python / 50)
        if len(X) >= 2:
            X_normalized = np.array([X[0], X[1] / 50.0])
        else:
            X_normalized = X
        
        # Линейная комбинация через скалярное произведение
        y = np.sum(X_normalized * self.B) + self.b0
        
        return float(y)
    
    def predict_class(self, X: np.ndarray) -> int:
        """
        Возвращает класс на основе порогового значения.
        
        Args:
            X: numpy массив с входными данными
            
        Returns:
            int: класс (0 или 1)
        """
        value = self.predict(X)
        return 1 if value > 0.5 else 0
    
    def predict_with_plot(self, X: np.ndarray) -> tuple[float, plt.Figure]:
        """
        Применяет модель линейной регрессии и создает график.
        
        Args:
            X: numpy массив с входными данными
            
        Returns:
            tuple: (предсказанное значение, matplotlib figure)
        """
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        
        prediction = self.predict(X)
        
        # Создание графика
        fig = plt.figure(figsize=(12, 5))
        
        # Первый подграфик - поверхность предсказаний
        ax1 = fig.add_subplot(121, projection='3d')
        
        # Генерация данных для визуализации
        x1_range = np.linspace(0, 100, 50)
        x2_range = np.linspace(0, 50, 50)
        X1, X2 = np.meshgrid(x1_range, x2_range)
        
        # Вычисление предсказаний для сетки с использованием NumPy операций
        X2_normalized = X2 / 50.0
        Y = self.b0 + self.B[0] * X1 + self.B[1] * X2_normalized
        
        # 3D поверхность предсказаний
        surf = ax1.plot_surface(X1, X2, Y, cmap='viridis', alpha=0.6)
        
        # Добавление точки предсказания
        ax1.scatter([X[0]], [X[1]], [prediction], 
                   color='red', s=100, label=f'Значение: {prediction:.3f}')
        
        ax1.set_xlabel('Оценка по ИС')
        ax1.set_ylabel('Баллы по Python')
        ax1.set_zlabel('Предсказанное значение')
        ax1.set_title('Линейная регрессия - Поверхность предсказаний')
        ax1.legend()
        
        # Второй подграфик - контурная карта с линией регрессии
        ax2 = fig.add_subplot(122)
        
        # Контурная карта
        contour = ax2.contourf(X1, X2, Y, levels=20, cmap='viridis')
        
        # Линия уровня y = 0.5 (граница классификации)
        ax2.contour(X1, X2, Y, levels=[0.5], colors='red', linewidths=2)
        
        # Добавление точки предсказания
        ax2.plot(X[0], X[1], 'ro', markersize=10, 
                label=f'Значение: {prediction:.3f}')
        
        ax2.set_xlabel('Оценка по ИС')
        ax2.set_ylabel('Баллы по Python')
        ax2.set_title('Линия регрессии (y=0.5)')
        ax2.legend()
        
        plt.colorbar(contour, ax=ax2, label='Предсказанное значение')
        plt.tight_layout()
        
        return prediction, fig
