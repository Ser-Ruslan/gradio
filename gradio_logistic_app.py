import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from logistic_regression import LogisticRegressionModel


class LogisticGradioServer:
    """
    Класс для запуска Gradio сервера с интерфейсом для логистической регрессии.
    """
    
    def __init__(self):
        """
        Инициализация сервера с моделью логистической регрессии.
        """
        self.model = LogisticRegressionModel()
    
    def predict_interface(self, is_score, python_balls):
        """
        Функция для обработки входных данных из Gradio интерфейса.
        
        Args:
            is_score: оценка по ИС
            python_balls: количество баллов по Python
            
        Returns:
            tuple: (результат предсказания, вероятность, график)
        """
        try:
            # Создание numpy массива из входных данных
            input_array = np.array([float(is_score), float(python_balls)])
            
            # Получение предсказания, вероятности и графика
            prediction = self.model.predict(input_array)
            probability = self.model.predict_proba(input_array)
            fig = self.model.predict_with_plot(input_array)[1]
            
            # Сохранение графика в виде изображения
            fig.savefig('logistic_plot.png', dpi=100, bbox_inches='tight')
            plt.close(fig)
            
            # Формирование текстовых результатов
            class_text = f"Предсказанный класс: {prediction}"
            prob_text = f"Вероятность класса 1: {probability:.4f}"
            
            # Интерпретация результата
            if prediction == 1:
                interpretation = "✅ Положительный результат (высокая вероятность успеха)"
            else:
                interpretation = "❌ Отрицательный результат (низкая вероятность успеха)"
            
            result_text = f"{class_text}\n{prob_text}\n{interpretation}"
            
            return result_text, probability, 'logistic_plot.png'
            
        except Exception as e:
            error_msg = f"Ошибка при обработке данных: {str(e)}"
            return error_msg, 0.0, None
    
    def create_interface(self):
        """
        Создание Gradio интерфейса.
        
        Returns:
            gr.Interface: настроенный интерфейс Gradio
        """
        # Создание интерфейса
        interface = gr.Interface(
            fn=self.predict_interface,
            inputs=[
                gr.Number(
                    label="Оценка по ИС (x1)",
                    value=25.0,
                    minimum=0,
                    maximum=100,
                    step=0.1
                ),
                gr.Number(
                    label="Баллы по Python (x2)",
                    value=25.0,
                    minimum=0,
                    maximum=50,
                    step=0.1
                )
            ],
            outputs=[
                gr.Textbox(label="Результат предсказания"),
                gr.Number(label="Вероятность класса 1", precision=4),
                gr.Image(label="График логистической регрессии")
            ],
            title="Сервер логистической регрессии",
            description=(
                "Модель логистической регрессии с коэффициентами:\n"
                "b0 = 48.6 (количество баллов по ИС)\n"
                "b1 = 2 (оценка по ИС)\n"
                "b2 = 45.9/50 (количество баллов по Python / 50)\n\n"
                "Формула: p = 1 / (1 + exp(-z))\n"
                "где z = b0 + b1*x1 + b2*x2\n\n"
                "Класс 1: вероятность > 0.5\n"
                "Класс 0: вероятность ≤ 0.5"
            ),
            examples=[
                [10.0, 10.0],  # Низкие показатели - класс 0
                [25.0, 25.0],  # Средние показатели
                [50.0, 40.0],  # Высокие показатели - класс 1
                [75.0, 45.0]   # Очень высокие показатели - класс 1
            ],
            allow_flagging="never"
        )
        
        return interface
    
    def run(self, server_name="127.0.0.1", server_port=7861, share=False):
        """
        Запуск Gradio сервера.
        
        Args:
            server_name: имя сервера
            server_port: порт сервера
            share: флаг публичного доступа
        """
        interface = self.create_interface()
        interface.launch(
            server_name=server_name,
            server_port=server_port,
            share=share,
            show_error=True
        )


def main():
    """
    Главная функция для запуска сервера.
    """
    server = LogisticGradioServer()
    server.run()


if __name__ == "__main__":
    main()
