import numpy as np


# Представление статичной картинки событиями atis
def mnist_image_to_events(
        image,      # изображение mnist (матрица 28 на 28)
        steps,      # сколько кадров
        direction,  # направление сдвига (right, left, up, down)
        dt,         # время между кадрами
        threshold   # порог изменения яркости для генерации события
):
    """
    
    Представляем статичную картинку mnist событиями atis:
    1)  смещаем картинку в направлении direction
    2)  пиксель генерирует событие, когда его яркость изменилась сильнее, чем на threshold
        одно событие: (t, x, y, polarity)
        t - момент времени, когда изменилась яркость пикселя
        x, y - координаты пикселя 
        polarity - ярче (1)/темнее (0)

    """
    events = []

    # Сохраняем текущий кадр
    curr_frame = image.copy()
    # Размер картинки
    N, M = image.shape

    # Создаем steps кадров
    for i in range(steps):
        # Сдвигаем картинку в заданном направлении
        next_frame = shift_image(curr_frame, direction)

        # Считаем изменение яркости пикселей
        diff = next_frame - curr_frame

        # Генерируем время события (номер кадра на время между кадрами)
        t = i * dt
        for y in range(N):
            for x in range(M):
                delta = diff[y, x]
                # Фиксируем изменение яркости, если изменение сильнее threshold
                if delta > threshold:
                    events.append((t, x, y, 1))  # рост яркости
                elif delta < -threshold:
                    events.append((t, x, y, 0))  # спад яркости

        # Обновляем кадр
        curr_frame = next_frame

    return events



# Вспомогательная функция для сдвига изображения
def shift_image(img, direction):

    # Создаем черный фон для незаполненных после сдвига пикселей
    shifted = np.zeros_like(img)

    # Первый срез по строкам, второй по столбцам
    if direction == "right":
        shifted[:, 1:] = img[:, :-1]
    elif direction == "left":
        shifted[:, :-1] = img[:, 1:]
    elif direction == "down":
        shifted[1:, :] = img[:-1, :]
    elif direction == "up":
        shifted[:-1, :] = img[1:, :]

    return shifted



# Визуализация для проверки

import matplotlib.pyplot as plt

def plot_event_raster(events, title="События ATIS"):
    """

    events: список событий
    title: заголовок (строка)

    """

    if not events:
        print("Список событий пуст")
        return

    # Разделим события по полярности
    x_on, y_on, t_on = [], [], []
    x_off, y_off, t_off = [], [], []

    for t, x, y, p in events:
        if p == 1:  # рост яркости
            x_on.append(x)
            y_on.append(y)
            t_on.append(t)
        else:       # спад яркости
            x_off.append(x)
            y_off.append(y)
            t_off.append(t)

    plt.figure()

    # Увеличение яркости отображается кружочком, цвет кодируется временем события
    plt.scatter(x_on, y_on, c=t_on, cmap='viridis', marker='o', s=20, label='1')

    # Спад яркости отображается крестиком, цвет кодируется временем события
    plt.scatter(x_off, y_off, c=t_off, cmap='viridis', marker='x', s=20, label='0')

    # Переворачиваем ось Y для корректного отображения изображения
    plt.gca().invert_yaxis()

    # Цветовая шкала по времени
    plt.colorbar(label="Время")

    plt.legend()
    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

