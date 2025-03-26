import matplotlib.pyplot as plt
import numpy as np


# Визуализирует список спайков в виде растра
def spike_raster(spikes):
    if not spikes:
        print("Сеть не выдала ни одного спайка")
        return

    times = [t for t, neuron_id in spikes]
    neuron_ids = [neuron_id for t, neuron_id in spikes]

    plt.figure()
    plt.scatter(times, neuron_ids, s=10, color='black') 
    plt.xlabel("Время (мкс)")
    plt.ylabel("Номер нейрона")
    plt.title("Растр спайков SNN")
    plt.grid(True)
    plt.tight_layout()
    plt.yticks(range(max(neuron_ids)+1))
    plt.show()



# Показывает веса заданного нейрона как две картинки: ON (1) и OFF (0)
def show_neuron_weights(weights, neuron_index):
    """
    
    weights: матрица весов [нейрон][вход]
    neuron_index: номер нейрона, которого хотим показать

    """

    # Получаем веса нужного нейрона (всего 1568 штук)
    w = weights[neuron_index]

    # Разделяем на ON и OFF
    w_on = w[::2]   # веса для полярности = 0 (OFF)
    w_off = w[1::2] # веса для полярности = 1 (ON)

    # Преобразуем в форму 28x28
    w_on_img = w_on.reshape((28, 28))
    w_off_img = w_off.reshape((28, 28))

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(w_on_img, cmap='viridis')
    plt.title(f"Нейрон {neuron_index} — OFF веса")
    plt.axis('off')
    plt.subplot(1, 2, 2)
    plt.imshow(w_off_img, cmap='viridis')
    plt.title(f"Нейрон {neuron_index} — ON веса")
    plt.axis('off')
    plt.tight_layout()
    plt.show()


# Строит график скользящего среднего точности по ходу обучения
def plot_learning(history, window=20):

    if len(history) < window:
        smoothed = history
    else:
        smoothed = [
            sum(history[i-window:i]) / window
            for i in range(window, len(history)+1)
        ]

    plt.figure()
    plt.plot(range(len(smoothed)), smoothed, label=f"Среднее window={window}")
    plt.title("Точность предсказания направления в процессе обучения")
    plt.xlabel("Шаг")
    plt.ylabel("Точность")
    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    plt.show()


# Показывает, сколько раз каждый нейрон сработал на каждое направление
def plot_neuron_stats(neuron_stats):
    directions = list(neuron_stats.keys())
    num_neurons = len(neuron_stats[directions[0]])

    x = list(range(num_neurons))
    width = 0.2
    offsets = [-1.5 * width, -0.5 * width, 0.5 * width, 1.5 * width]

    plt.figure(figsize=(12, 6))

    for i in range(len(directions)):
        d = directions[i]
        stats = neuron_stats[d]
        x_offset = [xi + offsets[i] for xi in x]
        plt.bar(x_offset, stats, width=width, label=d)

    plt.xlabel("Номер нейрона")
    plt.ylabel("Количество спайков")
    plt.title("Активность нейронов по направлениям")
    plt.xticks(x)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()