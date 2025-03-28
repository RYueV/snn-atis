import numpy as np

from visualization import plot_events, plot_time_histogram


# Сдвиг изображения в указанном направлении на долю пикселя
def shift_image_with_interpolation(
        frame,          # исходный кадр
        direction,      # направление сдвига (right, left, up, down)
        fraction        # на какую долю пикселя нужно выполнить сдвиг
):
    """

    Используется билинейная интерполяция.

    """
    # Получаем размер исходного кадра
    N, M = frame.shape
    # Создаем изображение с черным фоном для нового кадра
    new_frame = np.zeros_like(frame)

    # Обходим все строки
    for y in range(N):
        # И столбцы нового изображения
        for x in range(M):
            # Ищем новые координаты на исходном изображении
            if direction == 'right':    # если движемся вправо, то пиксель пришел слева
                new_x = x - fraction    # поэтому вычитаем смещение по горизонтальной оси
                new_y = y               # вертикальную ось не трогаем
            
            elif direction == 'left':   # если движемся влево, то пиксель пришел справа
                new_x = x + fraction    # добавляем смещение по горизонтальной оси
                new_y = y
            
            elif direction == 'down':   # если движемся вниз, то пиксель пришел сверху
                new_x = x
                new_y = y - fraction
            
            elif direction == 'up':     # если движемся вверх, то пиксель пришел снизу
                new_x = x
                new_y = y + fraction

            # Если направление указано неверно, то не меняем картинку
            else:
                new_x = x
                new_y = y

            # Ищем соседей нового пикселя на старом изображении
            # (ближайшие целые координаты)
            x0 = int(np.floor(new_x))   # левый
            x1 = x0 + 1                 # правый
            y0 = int(np.floor(new_y))   # верхний
            y1 = y0 + 1                 # нижний

            # Считаем относительные координаты
            dx = new_x - x0
            dy = new_y - y0

            # Инициализируем значения соседей (если выходим за границы - фон черный)
            top_left = 0.0          # левый верхний
            top_right = 0.0         # правый верхний
            bottom_left = 0.0       # левый нижний
            bottom_right = 0.0      # правый нижний

            # Проверяем границы
            if 0 <= x0 < M and 0 <= y0 < N:
                top_left = frame[y0, x0]
            if 0 <= x1 < M and 0 <= y0 < N:
                top_right = frame[y0, x1]
            if 0 <= x0 < M and 0 <= y1 < N:
                bottom_left = frame[y1, x0]
            if 0 <= x1 < M and 0 <= y1 < N:
                bottom_right = frame[y1, x1]

            # Формула билинейной интерполяции
            new_frame[y, x] = (
                top_left * (1 - dx) * (1 - dy) +
                top_right * dx * (1 - dy) +
                bottom_left * (1 - dx) * dy +
                bottom_right * dx * dy
            )

    return new_frame



# Генерирует поток событий из статичной картинки mnist
def mnist_image_to_events(
        image,              # исходная картинка (2d матрица 28x28)
        steps=10,           # количество сдвигов на целый пиксель в одном направлении
        direction='right',  # направление сдвига (right, left, up, down)
        dt=5.0,             # кол-во мс, за которое выполняется сдвиг на один пиксель
        threshold=0.67,     # порог изменения яркости для генерации события (логарифмическая шкала)
        substeps=2,         # за сколько шагов выполняется сдвиг на один пиксель
        pixel_ref=3.0,      # рефрактерный период для пикселя (в мс)
        visual=False,       # нужна ли визулизация (отображение гистограмм)
        ms_sub=1,           # на сколько бинов дробить 1 мс на гистограмме
):
    """

    Представляем статичную картинку mnist событиями atis:
    1)  смещаем картинку в направлении direction
    2)  пиксель генерирует событие, когда его яркость изменилась сильнее, чем на threshold
        |log(next) - log(curr)| >= threshold -> событие;
        одно событие: (t, x, y, polarity);
        t - момент времени, когда изменилась яркость пикселя;
        x, y - координаты пикселя;
        polarity - ярче (1)/темнее (0)

    На пиксель накладывается рефрактерный период: 
        пиксель не может генерировать события чаще, чем каждые pixel_ref мс;
        это нужно для "сглаживания" потока событий;
        при низком значении pixel_ref на первых шагах возникнет слишком много
        событий, из-за чего snn будет сложно правильно реагировать

    """
    # Размер изображения
    N, M = image.shape
    # Константа, чтобы избежать log(0)
    eps = 1e-5
    # Время последнего события для каждого пикселя и полярности
    last_event_time = np.full((N, M, 2), -np.inf)
    # Список событий
    events = []
    # Доля пикселя, на которую выполняется сдвиг за один подшаг
    fraction = 1.0 / substeps

    # Сохраняем текущий кадр
    curr_frame = image.copy()

    # Выполняем steps сдвигов на пиксель
    for i in range(steps):
        # Внутри каждого steps - substeps сдвигов на долю пикселя fraction
        for j in range(substeps):
            # Сдвигаем картинку в указанном направлении
            next_frame = shift_image_with_interpolation(curr_frame, direction, fraction)

            # Считаем логарифмическую разность
            diff_log = np.log(next_frame + eps) - np.log(curr_frame + eps)

            # Считаем интервал времени для подшага
            t_start = i * dt + j * (dt / substeps)
            t_end   = i * dt + (j + 1) * (dt / substeps)

            # Перебираем все пиксели
            for y in range(N):
                for x in range(M):
                    d = diff_log[y, x]
                    # Если разница достигла threshold, то генерируем событие
                    if abs(d) >= threshold:
                        # Если пиксель стал ярче, то полярность 1, иначе 0
                        p = 1 if d > 0 else 0

                        # Генерируем случайное время в пределах (t_start, t_end),
                        # чтобы избежать большого потока событий в одно время
                        t_event = np.random.uniform(t_start, t_end)

                        # Проверяем рефрактерный период
                        if t_event - last_event_time[y, x, p] >= pixel_ref:
                            events.append((t_event, x, y, p))
                            last_event_time[y, x, p] = t_event

            # Обновляем кадр
            curr_frame = next_frame

    # Сортируем события по времени наступления
    events.sort(key=lambda e: e[0])

    # Строим графики, если нужно
    if visual:
        plot_events(events)
        plot_time_histogram(events, ms_sub)

    return events
