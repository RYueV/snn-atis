import random

from encoding import *
import visualization as v


def train_epochs(
    snn,                                        # объект SNNnetwork
    mnist_dataset,                              # загруженный датасет mnist
    directions=("right", "down", "left", "up"), # направления для сдвига
    images_per_direction=200,                   # количество обрабатываемых картинок для каждого направления
    steps_per_image=10,                         # на сколько пикселей сдвигать картинку
    dt=5.0,                                     # время, за которое выполняется сдвиг на 1 пиксель (мс)
    threshold=0.67,                             # порог яркости для генерации события
    pixel_ref=3.0,                              # рефрактерный период для генерации событий (мс)
    substeps=2,                                 # на сколько шагов делится сдвиг на один пиксель
    epochs=3,                                   # количество эпох (сколько раз прогоняем на одной выборке)
    use_cached_events=True,                     # если true, то используем сохраненный датасет событий
    cache_path="data/cached_events.pkl"         # путь для сохранения сгенерированных событий
):
    """

    Обучение SNN по нескольким эпохам:
        1) заранее формирует события для всех картинок и направлений;
        2) перемешивает события на каждую эпоху, чтобы избежать переобучения
        3) в конце каждой эпохи пересчитывает предпочтения нейронов и точность предсказания
           (neuron_preference пересчитывается отдельно по каждой эпохе)

    """

    # Заранее переводим в события, чтобы не дублировать вычисления
    if use_cached_events:   # Если есть кэшированные события, то берем их
        all_events = binary_to_list(cache_path)
    else:   # Иначе генерируем 
        all_events = []
        # Берем первые images_per_direction картинок датасета
        for idx in range(images_per_direction):
            # Извлекаем тензор 
            # (метку не учитываем, т.к. нам не нужна информация, что за цифра на изображении)
            img_tensor, _ = mnist_dataset[idx]
            # Удаляем лишнюю ось и переводим в 2d матрицу
            image = img_tensor.squeeze().numpy()

            # Для каждой картинки 4 направления
            for direction_label in directions:
                # Формируем события
                events = mnist_image_to_events(
                    image=image,                # исходная картинка
                    direction=direction_label,  # направление сдвига
                    steps=steps_per_image,      # количество сдвигов на целый пиксель в одном направлении
                    substeps=substeps,          # за сколько шагов выполняется сдвиг на один пиксель
                    threshold=threshold,        # порог изменения яркости для генерации события (логарифмическая шкала)
                    pixel_ref=pixel_ref,        # рефрактерный период для пикселя (в мс)
                    dt=dt                       # кол-во мс, за которое выполняется сдвиг на один пиксель
                )
                # Храним пары (направление, список событий)
                all_events.append((direction_label, events))
        # Сохраняем список событий в файл "data/cached_events.pkl"
        list_to_binary(cache_path, all_events)


    # Обучаем на одной и той же выборке epochs раз
    for epoch in range(epochs):
        print(f"###  Эпоха {epoch + 1}/{epochs}  ###")

        # Перемешиваем выборку, чтобы избежать переобучения
        random.shuffle(all_events)

        # Количество спайков нейронов по направлениям (обновляется каждый раз)
        neuron_stats_epoch = {d: [0 for _ in range(snn.count_neurons)] for d in directions}

        # Статистика предсказаний
        correct_predictions = 0     # кол-во верных предсказаний
        total_predictions = 0       # общее кол-во предсказаний

        # Обрабатываем все пары
        for direction_label, events in all_events:
            # Сбрасываем состояние сети (кроме весов) перед новым изображением
            snn.reset()

            # Подаем события в сеть
            for t, x, y, p in events:
                snn.events_to_lif(t, x, y, p)

            # Подсчитываем количество спайков каждого нейрона
            spike_counts = [0] * snn.count_neurons
            for _, neuron_id in snn.spikes:
                spike_counts[neuron_id] += 1

            # Обновляем статистику спайков нейронов для текущего направления
            for neuron_id, count in enumerate(spike_counts):
                neuron_stats_epoch[direction_label][neuron_id] += count

            # Промежуточное вычисление предпочтений нейронов
            neuron_preference_epoch = {}
            # Обходим все нейроны
            for neuron_id in range(snn.count_neurons):
                # Создаем список спайков нейрона neuron_id для каждого направления
                counts = [neuron_stats_epoch[d][neuron_id] for d in directions]
                # Находим максимальное количество спайков нейрона neuron_id
                max_count = max(counts)
                # Находим направление, при котором нейрон neuron_id выдал 
                # максимальное количество спайков
                neuron_preference_epoch[neuron_id] = (
                    # Если не было ни одного спайка, то предпочитаемое направление отсутствует (None)
                    directions[counts.index(max_count)] if max_count > 0 else None
                )

            # Вычисляем промежуточное предсказание
            # Для этого создаем словарь "голосов" за каждое направление
            direction_votes = {d: 0 for d in directions}
            # Обходим пары (нейрон, количество спайков)
            for neuron_id, count in enumerate(spike_counts):
                # Из ранее собранной статистики узнаем, какое направление предпочитает нейрон
                pref = neuron_preference_epoch.get(neuron_id)
                # Если направление не None
                if pref:
                    # Прибавляем к этому направлению количество спайков этого нейрона
                    # (только те спайки, которые были получены при обработке данного направления)
                    direction_votes[pref] += count

            # Находим направление с максимальным количеством голосов за время обработки картинки
            predicted_direction = max(direction_votes, key=direction_votes.get)

            # Если предсказание верное
            if predicted_direction == direction_label:
                # Увеличиваем счетчик верных предсказаний
                correct_predictions += 1
            # Увеличиваем счетчик общего количества предсказаний
            total_predictions += 1

        # Считаем точность за эпоху
        epoch_accuracy = correct_predictions / total_predictions
        print(f"Точность за эпоху {epoch + 1}: {epoch_accuracy:.2%}")

        # Строим гистограмму предпочтений нейронов за эпоху
        v.plot_neuron_direction_stats(neuron_stats_epoch, directions)
        # Строим гистограммы весов для первых нейронов (для отладки)
        for neuron_id in range(5):
            v.plot_neuron_weight_histogram(snn.weights, neuron_id)


    # Возвращаем результаты за последнюю эпоху
    final_neuron_stats = neuron_stats_epoch
    final_neuron_preference = neuron_preference_epoch


    return final_neuron_stats, final_neuron_preference

