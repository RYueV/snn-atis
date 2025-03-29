from torchvision import datasets, transforms

from core import SNNnetwork
from visualization import plot_neuron_preferences
from encoding import list_to_binary
import train



if __name__ == "__main__":
    # Загружаем датасет
    # Каждый объект mnist - картинка (матрица) и метка (цифра на изображении)
    mnist = datasets.MNIST(
        root = "data/mnist",                   
        train = True,                       # обучающая выборка на 60000 картинок
        download = False,        
        transform = transforms.ToTensor()   # [1, 28, 28]; яркость нормирована 0...1
    )

    # Создаем сеть с 24 нейронами
    snn = SNNnetwork(count_neurons=24)

    # Обучаем сеть 
    # (одна картинка обрабатывается 50 мс, за это время происходит ~500-700 событий)
    _, neuron_preference, _ = train.train_epochs(
        snn=snn,                    # объект SNNnetwork
        mnist_dataset=mnist,        # загруженный датасет
        images_per_direction=200,   # кол-во картинок на одно направление
        steps_per_image=10,         # кол-во сдвигов на 1 пиксель в одном направлении
        dt=5.0,                     # время выполнения сдвига на 1 пиксель (мс)
        threshold=0.67,             # порог изменения яркости для генерации события (логарифмическая шкала)
        epochs=3,                   # кол-во эпох обучения (сколько раз прогоняем на одной и той же выборке)
        use_cached_events=True      # если True, используются кэшированные события, иначе генерируются новые
    )

    # Гистограмма предпочтений нейронов по направлениям после обучения
    plot_neuron_preferences(neuron_preference, ("right", "down", "left", "up"))

    # Сохраняем веса
    #list_to_binary("data/snn_weights.pkl", snn.weights)

    # Тестируем сеть после обучения (веса уже зафиксированы)
    test_history, test_accuracy = train.test_prediction(
        snn=snn,
        neuron_preference=neuron_preference,
        mnist_dataset=mnist,
        directions=("right", "down", "left", "up"),
        images_per_direction=100,
        steps_per_image=10,
        dt=5.0,
        threshold=0.67,
        start_index=800  
    )


