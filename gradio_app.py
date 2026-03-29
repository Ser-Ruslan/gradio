import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
from linear_regression import LinearRegressionModel


class GradioServer:
    """
    Класс для запуска Gradio сервера с интерфейсом для линейной регрессии.
    """
    
    def __init__(self):
        """
        Инициализация сервера с моделью линейной регрессии.
        """
        self.model = LinearRegressionModel()
    
    def predict_interface(self, is_score, python_balls):
        """
        Функция для обработки входных данных из Gradio интерфейса.
        
        Args:
            is_score: оценка по ИС
            python_balls: количество баллов по Python
            
        Returns:
            tuple: (результат предсказания, график)
        """
        try:
            # Создание numpy массива из входных данных
            input_array = np.array([float(is_score), float(python_balls)])
            
            # Получение предсказания и графика
            prediction, fig = self.model.predict_with_plot(input_array)
            
            # Сохранение графика в виде изображения
            fig.savefig('prediction_plot.png', dpi=100, bbox_inches='tight')
            plt.close(fig)
            
            # Формирование текстового результата
            result_text = f"Результат предсказания: {prediction}"
            
            return result_text, 'prediction_plot.png'
            
        except Exception as e:
            error_msg = f"Ошибка при обработке данных: {str(e)}"
            return error_msg, None
    
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
                gr.Image(label="График предсказания")
            ],
            title="Сервер линейной регрессии",
            description=(
                "Модель линейной регрессии с коэффициентами:\n"
                "b0 = 48.6 (количество баллов по ИС)\n"
                "b1 = 2 (оценка по ИС)\n"
                "b2 = 45.9/50 (количество баллов по Python / 50)\n\n"
                "Формула: y = b0 + b1*x1 + b2*x2"
            ),
            examples=[
                [2.0, 25.0],
                [3.5, 40.0],
                [1.0, 50.0]
            ],
            allow_flagging="never"
        )
        
        return interface
    
    def run(self, server_name="127.0.0.1", server_port=7860, share=False):
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
    server = GradioServer()
    server.run()


if __name__ == "__main__":
    main()
