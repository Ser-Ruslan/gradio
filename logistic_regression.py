import numpy as np
import matplotlib.pyplot as plt
from typing import Union


class LogisticRegressionModel:
    """
    Класс для имитации логистической регрессии с предопределенными коэффициентами.
    """
    
    def __init__(self):
        """
        Инициализация модели с коэффициентами регрессии.
        """
        # Коэффициенты регрессии для логистической регрессии
        # Используем более реалистичные коэффициенты для демонстрации
        self.b0 = -10.0  # свободный член (смещение)
        self.b1 = 0.2    # коэффициент для оценки по ИС
        self.b2 = 0.15   # коэффициент для баллов по Python
    
    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """
        Сигмоидная функция активации.
        
        Args:
            z: входной массив
            
        Returns:
            np.ndarray: результат применения сигмоиды
        """
        return 1 / (1 + np.exp(-z))
    
    def predict(self, input_array: np.ndarray) -> Union[float, int]:
        """
        Применяет модель логистической регрессии к входному массиву.
        
        Args:
            input_array: numpy массив с входными данными
            
        Returns:
            float или int: результат предсказания (вероятность или класс)
        """
        if not isinstance(input_array, np.ndarray):
            input_array = np.array(input_array)
        
        # Проверка размерности
        if len(input_array.shape) == 1:
            input_array = input_array.reshape(1, -1)
        
        # Линейная комбинация: z = b0 + b1*x1 + b2*x2
        z = self.b0 + self.b1 * input_array[:, 0] + self.b2 * input_array[:, 1]
        
        # Применение сигмоиды для получения вероятности
        probabilities = self._sigmoid(z)
        
        # Возвращаем первую вероятность
        probability = probabilities[0]
        
        # Если вероятность > 0.5, возвращаем класс 1, иначе класс 0
        if probability > 0.5:
            return 1
        else:
            return 0
    
    def predict_proba(self, input_array: np.ndarray) -> float:
        """
        Возвращает вероятность принадлежности к классу 1.
        
        Args:
            input_array: numpy массив с входными данными
            
        Returns:
            float: вероятность принадлежности к классу 1
        """
        if not isinstance(input_array, np.ndarray):
            input_array = np.array(input_array)
        
        # Проверка размерности
        if len(input_array.shape) == 1:
            input_array = input_array.reshape(1, -1)
        
        # Линейная комбинация
        z = self.b0 + self.b1 * input_array[:, 0] + self.b2 * input_array[:, 1]
        
        # Применение сигмоиды
        probabilities = self._sigmoid(z)
        
        return float(probabilities[0])
    
    def predict_with_plot(self, input_array: np.ndarray) -> tuple[Union[float, int], plt.Figure]:
        """
        Применяет модель логистической регрессии и создает график.
        
        Args:
            input_array: numpy массив с входными данными
            
        Returns:
            tuple: (результат предсказания, matplotlib figure)
        """
        prediction = self.predict(input_array)
        probability = self.predict_proba(input_array)
        
        # Создание графика
        fig = plt.figure(figsize=(12, 5))
        
        # Первый подграфик - поверхность вероятностей
        ax1 = fig.add_subplot(121, projection='3d')
        
        # Генерация данных для визуализации
        x1_range = np.linspace(0, 100, 50)
        x2_range = np.linspace(0, 50, 50)
        X1, X2 = np.meshgrid(x1_range, x2_range)
        
        # Вычисление вероятностей для сетки
        Z = self.b0 + self.b1 * X1 + self.b2 * X2
        Probabilities = self._sigmoid(Z)
        
        # 3D поверхность вероятностей
        surf = ax1.plot_surface(X1, X2, Probabilities, cmap='viridis', alpha=0.6)
        
        # Добавление точки предсказания
        ax1.scatter([input_array[0]], [input_array[1]], [probability], 
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
        ax2.plot(input_array[0], input_array[1], 'ro', markersize=10, 
                label=f'Класс: {prediction}')
        
        ax2.set_xlabel('Оценка по ИС')
        ax2.set_ylabel('Баллы по Python')
        ax2.set_title('Граница решения (p=0.5)')
        ax2.legend()
        
        plt.colorbar(contour, ax=ax2, label='Вероятность')
        plt.tight_layout()
        
        return prediction, fig
