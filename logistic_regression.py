import numpy as np
import matplotlib.pyplot as plt
from typing import Union


class LogisticRegressionModel:
    """
    Класс для имитации логистической регрессии с предопределенными коэффициентами.
    """
    
    def __init__(self, b0: float = 42, B: np.ndarray = None):
        """
        Инициализация модели с коэффициентами регрессии.
        
        Args:
            b0: свободный член 
            B: numpy массив весов признаков
        """
        self.b0 = b0  # свободный член
        if B is None:
            self.B = np.array([4, 40])  
        else:
            self.B = np.array(B)
    
    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """
        Сигмоидная функция активации.
        
        Args:
            z: входной массив
            
        Returns:
            np.ndarray: результат применения сигмоиды
        """
        return 1 / (1 + np.exp(-z))
    
    def predict(self, X: np.ndarray) -> float:
        """
        Применяет модель логистической регрессии к входному массиву.
        
        Args:
            X: numpy массив с входными данными
            
        Returns:
            float: вероятность принадлежности к классу 1
        """
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        
        # Нормализация второго признака (points_python / 50)
        if len(X) >= 2:
            X_normalized = np.array([X[0], X[1] / 50.0])
        else:
            X_normalized = X
        
        # Линейная комбинация через скалярное произведение
        z = np.sum(X_normalized * self.B) + self.b0
        
        # Сигмоида
        probability = 1 / (1 + np.exp(-z))
        
        return float(probability)
    
    def predict_class(self, X: np.ndarray) -> int:
        """
        Возвращает класс предсказания (0 или 1).
        
        Args:
            X: numpy массив с входными данными
            
        Returns:
            int: класс (0 или 1)
        """
        probability = self.predict(X)
        return 1 if probability > 0.5 else 0
    
    def predict_with_plot(self, X: np.ndarray) -> tuple[int, plt.Figure]:
        """
        Применяет модель логистической регрессии и создает график.
        
        Args:
            X: numpy массив с входными данными
            
        Returns:
            tuple: (класс предсказания, matplotlib figure)
        """
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        
        prediction = self.predict_class(X)
        probability = self.predict(X)
        
        # Создание графика
        fig = plt.figure(figsize=(12, 5))
        
        # Первый подграфик - поверхность вероятностей
        ax1 = fig.add_subplot(121, projection='3d')
        
        # Генерация данных для визуализации
        x1_range = np.linspace(0, 100, 50)
        x2_range = np.linspace(0, 50, 50)
        X1, X2 = np.meshgrid(x1_range, x2_range)
        
        # Вычисление вероятностей для сетки с использованием NumPy операций
        X2_normalized = X2 / 50.0
        Z = self.b0 + self.B[0] * X1 + self.B[1] * X2_normalized
        Probabilities = 1 / (1 + np.exp(-Z))
        
        # 3D поверхность вероятностей
        surf = ax1.plot_surface(X1, X2, Probabilities, cmap='viridis', alpha=0.6)
        
        # Добавление точки предсказания
        ax1.scatter([X[0]], [X[1]], [probability], 
                   color='red', s=100, label=f'Вероятность: {probability:.3f}')
        
        ax1.set_xlabel('Оценка по ИС')
        ax1.set_ylabel('Баллы по Python')
        ax1.set_zlabel('Вероятность')
        ax1.set_title('Логистическая регрессия - Поверхность вероятностей')
        ax1.legend()
        ax1.set_zlim([0, 1])
        
        # Второй подграфик - контурная карта с границей решения
        ax2 = fig.add_subplot(122)
        
        # Контурная карта
        contour = ax2.contourf(X1, X2, Probabilities, levels=20, cmap='viridis')
        ax2.contour(X1, X2, Probabilities, levels=[0.5], colors='red', linewidths=2)
        
        # Добавление точки предсказания
        ax2.plot(X[0], X[1], 'ro', markersize=10, 
                label=f'Класс: {prediction}')
        
        ax2.set_xlabel('Оценка по ИС')
        ax2.set_ylabel('Баллы по Python')
        ax2.set_title('Граница решения (p=0.5)')
        ax2.legend()
        
        plt.colorbar(contour, ax=ax2, label='Вероятность')
        plt.tight_layout()
        
        return prediction, fig
