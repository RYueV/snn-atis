import numpy as np
from .LIFneuron import LIF_neuron_event

# Фиксируем random
np.random.seed(42)

class SNNnetwork:
    def __init__(
            self,
            count_neurons,          # количество нейронов в сети
            input2d_size=(28,28),   # размер входной матрицы
            t_inhibit=1.80445,      # время ингибирования (мс)
            t_ltp = 7.9024,         # окно "отклика" нейрона на входной сигнал (мс)
            alpha_plus=19.8231,     # инкремент веса связи
            alpha_minus=17.7124,    # декремент веса связи
            beta_plus=0.06644,      # инкремент усиления веса связи
            beta_minus=0.22925,     # декремент ослабления веса связи
    ):
        self.count_neurons = count_neurons
        self.t_inhibit = t_inhibit
        # Количество возможных входов (количество пикселей + 2 состояния на каждый пиксель)
        self.input_size = input2d_size[0] * input2d_size[1] * 2

        # Параметры STDP
        self.alpha_plus = alpha_plus
        self.alpha_minus = alpha_minus
        self.beta_plus = beta_plus
        self.beta_minus = beta_minus
        self.t_ltp = t_ltp

        # Матрица весов (строки - нейроны, веса - входы)
        self.weights = np.random.normal(317.5153, 47.0, size=(24, 28*28*2))
        self.weights = np.clip(self.weights, 20, 600)

        # Время последнего входа на каждый input_id
        self.last_input_times = np.zeros(self.input_size)
        # Создаем LIF нейроны
        self.neurons = [LIF_neuron_event(neuron_id=i) for i in range(count_neurons)]
        # Список спайков с моментами времени 
        self.spikes = []

        # Списки количества усилений (LTP) и ослаблений (LTD) связей на шаге
        self.ltp_counts_history = []  
        self.ltd_counts_history = [] 
        # Динамика весов (для отладки)
        self.weight_history = []


    # Сброс состояния сети
    def reset(self):
        self.last_input_times = np.zeros(self.input_size) 
        self.spikes = []
        for neuron in self.neurons:
            neuron.reset()
    

    # Передача ATIS событий в LIF нейроны
    def events_to_lif(self, t, x, y, p, train=True):
        # Номер входа
        input_id = 2 * (y * 28 + x) + p
        # Регистрируем время входного сигнала
        self.last_input_times[input_id] = t
        # Обходим список нейронов
        for i in range(self.count_neurons):
            neuron = self.neurons[i]
            # Достаем вес связи текущего нейрона со входом от atis
            weight = self.weights[i][input_id]
            # Выполняем один шаг симуляции LIF
            spiked, _ = neuron.process_input(t, weight)
            # Регистрируем спайк
            if spiked:
                self.spikes.append((t, i))
                self.inhibition(t, i)
                if train:
                    self.update_weights_stdp(t, i)


    # Ингибирование всех нейронов, кроме сработавшего
    def inhibition(self, t, spiked_id):
        inhibit_until = t + self.t_inhibit
        for i in range(self.count_neurons):
            if i == spiked_id:
                continue
            neuron = self.neurons[i]
            neuron.inhibited_until = max(neuron.inhibited_until, inhibit_until)


    # Обучение по правилу STDP
    def update_weights_stdp(self, t_post, neuron_index):
        """
        
        neuron_index: какой нейрон спайковал
        t_post: время спайка нейрона neuron_index

        """
        w_min = 20.0
        w_max = 600.0
        w_range = w_max - w_min

        # Счетчики ltp/ltd на текущем шаге
        ltp_count = 0
        ltd_count = 0

        # Идем по всем входам
        for input_id in range(self.input_size):
            # Время, когда пришел входной сигнал
            t_pre = self.last_input_times[input_id]
            # Разница времени между тем моментом, когда пришел входной сигнал
            # и моментом, когда отработал сам нейрон input_id
            delta_t = t_post - t_pre

            w = self.weights[neuron_index][input_id]

            # Если входной сигнал пришел до того момента, как нейрон отработал
            if 0 < delta_t < self.t_ltp:
                # Увеличиваем связь
                delta_w = self.alpha_plus * np.exp(-self.beta_plus * (w - w_min) / w_range)
                w += delta_w
                ltp_count += 1
            else:
                # Иначе ослабляем
                delta_w = self.alpha_minus * np.exp(-self.beta_minus * (w_max - w) / w_range)
                w -= delta_w
                ltd_count += 1

            # Ограничиваем вес
            self.weights[neuron_index][input_id] = np.clip(w, w_min, w_max)
        
        # Храним общую историю усилений и ослаблений, 
        # чтобы понимать, какие события происходят чаще 
        # и как лучше обновлять веса
        self.ltp_counts_history.append(ltp_count)
        self.ltd_counts_history.append(ltd_count)

        