import numpy as np
from core import SNNnetwork
from train import train_epochs


# Обучение snn на заданных параметрах
# Возвращает точность предсказаний за последнюю эпоху
def evaluate_snn_accuracy(
        params,                         # словарь параметров STDP, LIF, SNN
        mnist_dataset,                  # загруженный датасет mnist
        images_per_direction=50,        # количество обрабатываемых картинок для каждого направления
        epochs=1                        # количество эпох для тестирования гиперпараметров
):
    # Инициализация SNN заданными параметрами
    snn = SNNnetwork(
        count_neurons=24,                   # кол-во нейронов фиксировано
        alpha_plus=params["alpha_plus"],
        alpha_minus=params["alpha_minus"],
        beta_plus=params["beta_plus"],
        beta_minus=params["beta_minus"],
        t_inhibit=params["t_inhibit"],
        t_ltp=params["t_ltp"],
    )

    # Меняем стартовые веса связей на заданные
    snn.weights = np.random.normal(
        params["w_init_mean"],
        params["w_init_std"],
        size=(snn.count_neurons, 28*28*2)
    )
    snn.weights = np.clip(snn.weights, params["w_min"], params["w_max"])

    # Меняем значения параметров нейронов сети на заданные
    for neuron in snn.neurons:
        neuron.tau_leak = params["tau_leak"]
        neuron.I_thres = params["I_thres"]
        neuron.t_ref = params["t_ref"]


    # Запускаем обучение, фиксируем точность предсказаний за эхпоху
    _, _, accuracy = train_epochs(
        snn,
        mnist_dataset,
        images_per_direction=images_per_direction,
        # Используем фиксированные настройки генерации потока событий
        steps_per_image=10,
        dt=5.0,
        threshold=0.67,
        epochs=epochs,
        substeps=2,
        # Используем кэшированную выборку событий при каждом запуске,
        # внутри train_epochs эта выборка перемешивается
        use_cached_events=True                    
    )

    # Возвращаем точность предсказаний за последнюю эпоху
    return accuracy
