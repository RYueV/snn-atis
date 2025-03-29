import matplotlib.pyplot as plt
import numpy as np




#################################    Для настройки порога яркости   #####################################


# Строит графики распределения событий
def plot_events(events, time_bins=40):
    """

    events: список событий [(t_1, x_1, y_1, p_1), ..., (t_n, x_n, y_n, p_n)]
    time_bins: на какое количество бинов разбиваем

    Строит:
        1) гистограмму по времени (сколько событий в каждом промежутке)
        2) диаграммы рассеяния (t vs x) и (t vs y) с окраской по полярности:
            -- синий: пиксель стал ярче (полярность p=1)
            -- красный: пиксель стал темнее (полярность p=0)

    """
    times = [e[0] for e in events]
    xs = [e[1] for e in events]
    ys = [e[2] for e in events]
    ps = [e[3] for e in events]

    plt.figure()

    # Гистограмма времени
    plt.subplot(1, 3, 1)
    plt.hist(times, bins=time_bins)
    plt.title("Распределение событий по времени")
    plt.xlabel("Время (мс)")
    plt.ylabel("Количество событий")

    # Диаграмма рассеяния (t vs x)
    plt.subplot(1, 3, 2)
    plt.scatter(times, xs, c=ps, cmap='bwr', s=5, alpha=0.7)
    plt.xlabel("Время (мс)")
    plt.ylabel("X")

    # Диаграмма рассеяния (t vs y)
    plt.subplot(1, 3, 3)
    plt.scatter(times, ys, c=ps, cmap='bwr', s=5, alpha=0.7)
    plt.xlabel("Время (мс)")
    plt.ylabel("Y")

    plt.tight_layout()
    plt.show()


# Детализированная гистограмма распределения событий по времени
def plot_time_histogram(events, ms_sub=1):
    """

    events: список событий [(t_1, x_1, y_1, p_1), ..., (t_n, x_n, y_n, p_n)]
    ms_sub: на какое количество бинов разбивается 1 мс

    """
    times = [evt[0] for evt in events]

    t_min = min(times)
    t_max = max(times)
    bin_width = 1.0 / ms_sub
    bins = np.arange(t_min, t_max + bin_width, bin_width)

    plt.figure()
    plt.hist(times, bins=bins, color='blue', edgecolor='black', alpha=0.7)
    plt.xlabel("Время (мс)")
    plt.ylabel("Количество событий")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()



#################################    Визуализация результата обучения   #####################################


# Визуализация предпочтений нейронов
def plot_neuron_preferences(neuron_preference, directions):
    pref_counts = {d: 0 for d in directions}
    none_pref = 0
    for pref in neuron_preference.values():
        if pref is None:
            none_pref += 1
        else:
            pref_counts[pref] += 1

    labels = list(pref_counts.keys()) + ['None']
    counts = list(pref_counts.values()) + [none_pref]

    plt.figure()
    plt.bar(labels, counts, alpha=0.7)
    plt.title('Количество нейронов, предпочитающих каждое направление')
    plt.xlabel('Направление')
    plt.ylabel('Количество нейронов')
    plt.grid(True)
    plt.tight_layout()
    plt.show()



# Распределение спайков по нейронам и направлениям
def plot_neuron_direction_stats(neuron_stats, directions):

    num_neurons = len(next(iter(neuron_stats.values())))
    plt.figure()

    for neuron_id in range(num_neurons):
        counts = [neuron_stats[d][neuron_id] for d in directions]
        plt.bar([f"neuron {neuron_id} {d}" for d in directions], counts, alpha=0.7)

    plt.xticks(rotation=90)
    plt.ylabel('Количество спайков')
    plt.title('Распределение спайков нейронов по направлениям')
    plt.tight_layout()
    plt.show()




# Строит график скользящего среднего точности
def plot_learning(history, window=20):
    """

    history: список 0 или 1 (1 - верное предсказание, 0 - неверное)
    window: размер окна сглаживания

    """
    # Если история короче окна, просто рисуем точные значения
    if len(history) < window:
        smoothed = [sum(history[:i+1])/(i+1) for i in range(len(history))]
    else:
        smoothed = []
        for i in range(len(history)):
            start = max(0, i - window + 1)
            segment = history[start:i+1]
            avg = sum(segment) / len(segment)
            smoothed.append(avg)

    plt.figure()
    plt.plot(range(len(smoothed)), smoothed, label=f"скользящее среднее (окно={window})")
    plt.title("Точность предсказания в процессе обучения/тестирования")
    plt.xlabel("Шаг")
    plt.ylabel("Точность")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()



#################################    Для подбора весов   #####################################


def plot_weight_dynamics(weight_history, neuron_id, input_id):
    """
    Показывает динамику изменения веса связи нейронов к конкретному входу.
    weight_history — список весов по шагам (время можно аппроксимировать по индексу).
    """
    plt.figure(figsize=(10, 4))
    plt.plot(weight_history, marker='o')
    plt.xlabel("Шаг")
    plt.ylabel("Вес")
    plt.title(f"Динамика веса: Нейрон {neuron_id}, вход {input_id}")
    plt.grid(True)
    plt.tight_layout()
    plt.show()



# Изменение весов отдельных нейронов
def plot_neuron_weight_histogram(weights, neuron_id):
    plt.figure()
    plt.hist(weights[neuron_id], bins=30, alpha=0.7)
    plt.title(f'Распределение весов нейрона {neuron_id}')
    plt.xlabel('Вес')
    plt.ylabel('Количество связей')
    plt.grid(True)
    plt.tight_layout()
    plt.show()




#################################    Для подбора других гиперпараметров   #####################################


def plot_ltp_ltd_counts(ltp_counts, ltd_counts):
    """
    Строит график сравнения количества LTP и LTD обновлений весов на каждом шаге.
    """
    steps = list(range(len(ltp_counts)))
    plt.figure(figsize=(10, 4))
    plt.plot(steps, ltp_counts, label="LTP", color="green")
    plt.plot(steps, ltd_counts, label="LTD", color="red")
    plt.xlabel("Шаг")
    plt.ylabel("Количество обновлений")
    plt.title("Количество LTP и LTD обновлений по шагам")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()



# Визуализирует список спайков в виде растра
def spike_raster(spikes):
    """
    
    Принимает список пар (время спайка, номер нейрона).

    """
    if not spikes:
        print("Сеть не выдала ни одного спайка")
        return

    times = [t for t, _ in spikes]
    neuron_ids = [neuron_id for _, neuron_id in spikes]

    plt.figure()
    plt.scatter(times, neuron_ids, s=10, color='black') 
    plt.xlabel("Время (мс)")
    plt.ylabel("Номер нейрона")
    plt.title("Растр спайков SNN")
    plt.grid(True)
    plt.tight_layout()
    plt.yticks(range(max(neuron_ids)+1))
    plt.show()
