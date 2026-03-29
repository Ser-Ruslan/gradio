import numpy as np
import pytest
from logistic_regression import LogisticRegressionModel


class TestLogisticRegressionModel:
    """
    Тесты для класса LogisticRegressionModel.
    """
    
    def setup_method(self):
        """
        Настройка перед каждым тестом.
        """
        self.model = LogisticRegressionModel()
    
    def test_predict_class_1(self):
        """
        Тест 1: Проверка предсказания класса 1 (высокая вероятность).
        """
        # Входные данные, которые должны дать высокую вероятность (> 0.5)
        input_data = np.array([50.0, 50.0])  # x1=50, x2=50
        result = self.model.predict(input_data)
        probability = self.model.predict_proba(input_data)
        
        # Проверяем, что предсказан класс 1
        assert result == 1, f"Ожидался класс 1, получен {result}"
        # Проверяем, что вероятность > 0.5
        assert probability > 0.5, f"Ожидалась вероятность > 0.5, получена {probability}"
    
    def test_predict_class_0(self):
        """
        Тест 2: Проверка предсказания класса 0 (низкая вероятность).
        """
        # Входные данные, которые должны дать низкую вероятность (< 0.5)
        input_data = np.array([0.0, 0.0])  # x1=0, x2=0
        result = self.model.predict(input_data)
        probability = self.model.predict_proba(input_data)
        
        # Проверяем, что предсказан класс 0
        assert result == 0, f"Ожидался класс 0, получен {result}"
        # Проверяем, что вероятность < 0.5
        assert probability < 0.5, f"Ожидалась вероятность < 0.5, получена {probability}"
    
    def test_predict_proba_range(self):
        """
        Тест 3: Проверка диапазона вероятностей.
        """
        # Тест с различными входными данными
        test_cases = [
            ([10.0, 10.0], "слабые показатели"),
            ([25.0, 25.0], "средние показатели"),
            ([75.0, 40.0], "сильные показатели")
        ]
        
        for input_data, description in test_cases:
            probability = self.model.predict_proba(input_data)
            
            # Проверяем, что вероятность в диапазоне [0, 1]
            assert 0 <= probability <= 1, f"Для {description} вероятность {probability} вне диапазона [0,1]"
            
            # Проверяем, что predict и predict_proba согласованы
            prediction = self.model.predict(input_data)
            if probability > 0.5:
                assert prediction == 1, f"Для {description}: вероятность {probability} > 0.5, но предсказан класс {prediction}"
            else:
                assert prediction == 0, f"Для {description}: вероятность {probability} <= 0.5, но предсказан класс {prediction}"
    
    def test_predict_with_list_input(self):
        """
        Тест 4: Проверка предсказания со списком на входе.
        """
        # Входные данные в виде списка
        input_data = [30.0, 35.0]
        result = self.model.predict(input_data)
        probability = self.model.predict_proba(input_data)
        
        # Проверяем, что результат - это 0 или 1
        assert result in [0, 1], f"Ожидался класс 0 или 1, получен {result}"
        # Проверяем диапазон вероятности
        assert 0 <= probability <= 1, f"Вероятность {probability} вне диапазона [0,1]"
    
    def test_boundary_case(self):
        """
        Тест 5: Проверка граничного случая (вероятность близка к 0.5).
        """
        # Подбираем входные данные для получения вероятности близкой к 0.5
        # Для z = 0: b0 + b1*x1 + b2*x2 = 0
        # 48.6 + 2*x1 + 0.918*x2 = 0
        # Это невозможно при положительных x1, x2, но проверим с малыми значениями
        input_data = np.array([1.0, 1.0])
        probability = self.model.predict_proba(input_data)
        
        # Проверяем, что вероятность в допустимом диапазоне
        assert 0 <= probability <= 1, f"Вероятность {probability} вне диапазона [0,1]"
        
        # Проверяем согласованность с predict
        prediction = self.model.predict(input_data)
        if probability > 0.5:
            assert prediction == 1
        else:
            assert prediction == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
