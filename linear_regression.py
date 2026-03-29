import numpy as np
import matplotlib.pyplot as plt
from typing import Union


class LinearRegressionModel:
    """
    Класс для имитации линейной регрессии с предопределенными коэффициентами.
    """
    
    def __init__(self):
        """
        Инициализация модели с коэффициентами регрессии.
        """
        # Коэффициенты регрессии согласно заданию
        self.b0 = 48.6  # количество баллов по ИС
        self.b1 = 2     # оценка по ИС
        self.b2 = 45.9 / 50  # количество баллов по Python / 50
    
    def predict(self, input_array: np.ndarray) -> Union[float, int]:
        """
        Применяет модель линейной регрессии к входному массиву.
        
        Args:
            input_array: numpy массив с входными данными
            
        Returns:
            float или int: результат предсказания
        """
        if not isinstance(input_array, np.ndarray):
            input_array = np.array(input_array)
        
        # Проверка размерности
        if len(input_array.shape) == 1:
            input_array = input_array.reshape(1, -1)
        
        # Линейная регрессия: y = b0 + b1*x1 + b2*x2
        predictions = self.b0 + self.b1 * input_array[:, 0] + self.b2 * input_array[:, 1]
        
        # Возвращаем первый результат (или единственный результат)
        result = predictions[0]
        
        # Если результат близок к целому числу, возвращаем int
        if abs(result - round(result)) < 1e-10:
            return int(round(result))
        return float(result)
    
    def predict_with_plot(self, input_array: np.ndarray) -> tuple[Union[float, int], plt.Figure]:
        """
        Применяет модель линейной регрессии и создает график.
        
        Args:
            input_array: numpy массив с входными данными
            
        Returns:
            tuple: (результат предсказания, matplotlib figure)
        """
        prediction = self.predict(input_array)
        
        # Создание графика
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Генерация данных для визуализации
        x1_range = np.linspace(0, 100, 50)
        x2_range = np.linspace(0, 50, 50)
        X1, X2 = np.meshgrid(x1_range, x2_range)
        
        # Вычисление предсказаний для сетки
        Y = self.b0 + self.b1 * X1 + self.b2 * X2
        
        # 3D поверхность
        from mpl_toolkits.mplot3d import Axes3D
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X1, X2, Y, cmap='viridis', alpha=0.6)
        
        # Добавление точки предсказания
        ax.scatter([input_array[0]], [input_array[1]], [prediction], 
                  color='red', s=100, label=f'Предсказание: {prediction}')
        
        ax.set_xlabel('Оценка по ИС')
        ax.set_ylabel('Баллы по Python')
        ax.set_zlabel('Предсказание')
        ax.set_title('Линейная регрессия')
        ax.legend()
        
        plt.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        
        return prediction, fig
