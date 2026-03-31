import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from logistic_regression import LinearRegressionModel

# python -r requirements.txt
# python gradio_logistic_app.py
# python test_logistic_regression.py


class LinearGradioServer:
    """
    Класс для запуска Gradio сервера с интерфейсом для линейной регрессии.
    """
    
    def __init__(self):
        """
        Инициализация сервера с моделью линейной регрессии.
        """
        # Используем более сбалансированные коэффициенты
        self.model = LinearRegressionModel(b0=-10, B=np.array([0.2, 0.8]))
    
    def predict_interface(self, is_score, python_balls):
        """
        Функция для обработки входных данных из Gradio интерфейса.
        
        Args:
            is_score: оценка по ИС
            python_balls: количество баллов по Python
            
        Returns:
            tuple: (результат предсказания, значение, график)
        """
        try:
            # Создание numpy массива из входных данных
            X = np.array([float(is_score), float(python_balls)])
            
            # Получение предсказания, значения и графика
            prediction = self.model.predict_class(X)
            value = self.model.predict(X)
            fig = self.model.predict_with_plot(X)[1]
            
            # Сохранение графика в виде изображения
            fig.savefig('linear_plot.png', dpi=100, bbox_inches='tight')
            plt.close(fig)
            
            # Формирование текстовых результатов
            class_text = f"Предсказанный класс: {prediction}"
            value_text = f"Предсказанное значение: {value:.4f}"
            
            # Интерпретация результата
            if prediction == 1:
                interpretation = "Положительный результат"
            else:
                interpretation = "Отрицательный результат"
            
            result_text = f"{class_text}\n{value_text}\n{interpretation}"
            
            return result_text, value, 'linear_plot.png'
            
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
                    value=2.0,
                    minimum=0,
                    maximum=1000,
                    step=0.1
                ),
                gr.Number(
                    label="Баллы по Python (x2)",
                    value=48.0,
                    minimum=0,
                    maximum=1000,
                    step=0.1
                )
            ],
            outputs=[
                gr.Textbox(label="Результат предсказания"),
                gr.Number(label="Предсказанное значение", precision=4),
                gr.Image(label="График линейной регрессии")
            ],
            title="Сервер линейной регрессии",
            description=(
                "Модель линейной регрессии с коэффициентами:\n"
                "b0 = -10 (свободный член)\n"
                "B = [0.2, 0.8] (веса для [score_IS, points_python/50])\n\n"
                "Формула: y = b0 + Σ(Bi * Xi)\n\n"
                "Класс 1: значение > 0.5\n"
                "Класс 0: значение ≤ 0.5"
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
    server = LinearGradioServer()
    server.run()


if __name__ == "__main__":
    main()
