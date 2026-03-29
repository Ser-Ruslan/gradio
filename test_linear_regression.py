import numpy as np
import pytest
from linear_regression import LinearRegressionModel


class TestLinearRegressionModel:
    """
    Тесты для класса LinearRegressionModel.
    """
    
    def setup_method(self):
        """
        Настройка перед каждым тестом.
        """
        self.model = LinearRegressionModel()
    
    def test_predict_with_integer_result(self):
        """
        Тест 1: Проверка предсказания, которое должно вернуть целое число.
        """
        # Входные данные, которые должны дать целочисленный результат
        input_data = np.array([1.0, 50.0])  # x1=1, x2=50
        result = self.model.predict(input_data)
        
        # Ожидаемый результат: 48.6 + 2*1 + (45.9/50)*50 = 48.6 + 2 + 45.9 = 96.5
        expected = 96.5
        assert abs(result - expected) < 1e-10, f"Ожидалось {expected}, получено {result}"
    
    def test_predict_with_float_result(self):
        """
        Тест 2: Проверка предсказания, которое должно вернуть дробное число.
        """
        # Входные данные, которые должны дать дробный результат
        input_data = np.array([2.5, 25.0])  # x1=2.5, x2=25
        result = self.model.predict(input_data)
        
        # Ожидаемый результат: 48.6 + 2*2.5 + (45.9/50)*25 = 48.6 + 5 + 22.95 = 76.55
        expected = 76.55
        assert abs(result - expected) < 1e-10, f"Ожидалось {expected}, получено {result}"
        assert isinstance(result, float), f"Ожидался float, получен {type(result)}"
    
    def test_predict_with_list_input(self):
        """
        Тест 3: Проверка предсказания со списком на входе (не numpy массив).
        """
        # Входные данные в виде списка
        input_data = [0.0, 0.0]  # x1=0, x2=0
        result = self.model.predict(input_data)
        
        # Ожидаемый результат: 48.6 + 2*0 + (45.9/50)*0 = 48.6
        expected = 48.6
        assert abs(result - expected) < 1e-10, f"Ожидалось {expected}, получено {result}"
    
    def test_predict_with_single_array(self):
        """
        Тест 4: Дополнительный тест - проверка с одномерным массивом.
        """
        input_data = np.array([3.0, 10.0])
        result = self.model.predict(input_data)
        
        # Ожидаемый результат: 48.6 + 2*3 + (45.9/50)*10 = 48.6 + 6 + 9.18 = 63.78
        expected = 63.78
        assert abs(result - expected) < 1e-10, f"Ожидалось {expected}, получено {result}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
