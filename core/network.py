import numpy as np
from .LIFneuron import LIF_neuron_event


class SNNnetwork:
    def __init__(
            self,
            count_neurons,          # количество нейронов в выходном слое
            input2d_size=(28,28),   # размер входной матрицы
            t_inhibit=1.5,          # время ингибирования
            alpha_plus=50,         # инкремент веса связи
            alpha_minus=25,         # декремент веса связи
            beta_plus=0.0,          # инкремент усиления веса связи
            beta_minus=0.0,         # декремент ослабления веса связи
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
        # Временной масштаб работы единичного нейрона
        self.t_ltp = 2.5
        # Матрица весов (строки - нейроны, веса - входы)
        # self.weights = np.random.normal(loc=800, scale=53.33, size=(count_neurons, self.input_size))
        # self.weights = np.clip(self.weights, 640, 960) 
        self.weights = np.random.uniform(200, 300, size=(count_neurons, self.input_size))
        # self.weights = np.random.normal(loc=250, scale=52, size=(count_neurons, self.input_size))
        # self.weights = np.clip(self.weights, 50, 450)

        # Время последнего входа на каждый input_id
        self.last_input_times = np.zeros(self.input_size)
        # Создаем LIF нейроны
        self.neurons = [LIF_neuron_event(neuron_id=i) for i in range(count_neurons)]
        # Список спайков с моментами времени 
        self.spikes = []


    def reset(self):
        self.last_input_times = np.zeros(self.input_size) 
        self.spikes = []
        for neuron in self.neurons:
            neuron.reset()
    

    # Передача ATIS событий в LIF нейроны
    def events_to_lif(self, t, x, y, p):
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
        w_min = 1.0
        w_max = 1200.0
        w_range = w_max - w_min

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
                #self.weights[neuron_index][input_id] += self.alpha_plus
            else:
                # Иначе ослабляем
                delta_w = self.alpha_minus * np.exp(-self.beta_minus * (w_max - w) / w_range)
                w -= delta_w
                #self.weights[neuron_index][input_id] -= self.alpha_minus

            # Ограничиваем вес
            self.weights[neuron_index][input_id] = np.clip(w, w_min, w_max)





# from .LIFneuron import LIF_neuron

# # Использует модель LIF нейрона по шагам времени
# class SNNnetwork_old:
#     def __init__(
#             self,
#             count_neurons,          # количество нейронов в выходном слое
#             input2d_size=(28,28),   # размер входной матрицы
#             dt=1.0,                 # шаг интегрирования
#             t_inhibit=1.5,          # время ингибирования
#     ):
#         self.count_neurons = count_neurons
#         self.dt = dt
#         self.t_inhibit = t_inhibit
#         # Количество возможных входов (количество пикселей + 2 состояния на каждый пиксель)
#         self.input_size = input2d_size[0] * input2d_size[1] * 2
#         # Время последнего входа на каждый input_id
#         self.last_input_times = np.zeros(self.input_size)
#         # Матрица весов (строки - нейроны, веса - входы)
#         self.weights = np.random.normal(loc=800, scale=160, size=(count_neurons, self.input_size))
#         self.weights = np.clip(self.weights, 1, 1200)  # обрезка в границах ω_min и ω_max
#         #self.weights = np.random.uniform(0, 1, size=(count_neurons, self.input_size))
#         # Создаем LIF нейроны
#         self.neurons = [LIF_neuron(dt=dt, neuron_id=i) for i in range(count_neurons)]
#         # Список спайков с моментами времени 
#         self.spikes = []


#     def reset(self):
#         self.spikes = []
#         for neuron in self.neurons:
#             neuron.reset()
    

#     # Передача ATIS событий в LIF нейроны
#     def events_to_lif(self, t, x, y, p):
#         # Номер входа
#         input_id = 2 * (y * 28 + x) + p
#         # Регистрируем время входного сигнала
#         self.last_input_times[input_id] = t
#         # Обходим список нейронов
#         for i in range(self.count_neurons):
#             neuron = self.neurons[i]
#             # Достаем вес связи текущего нейрона со входом от atis
#             weight = self.weights[i][input_id]
#             # Выполняем один шаг симуляции LIF
#             spiked, _ = neuron.step(weight)
#             # Регистрируем спайк
#             if spiked:
#                 self.spikes.append((t, i))
#                 self.inhibition(t, i)
#                 self.update_weights_stdp(t, i)


#     # Ингибирование всех нейронов, кроме сработавшего
#     def inhibition(self, t, spiked_id):
#         inhibit_until = t + self.t_inhibit
#         for i in range(self.count_neurons):
#             if i == spiked_id:
#                 continue
#             neuron = self.neurons[i]
#             neuron.inhibited_until = max(neuron.inhibited_until, inhibit_until)


#     # Обучение по правилу STDP
#     def update_weights_stdp(self, t_spike, neuron_index):
#         """
        
#         neuron_index: какой нейрон спайковал
#         t_spike: время спайка нейрона

#         Значения констант взяты из таблицы.

#         """
#         # Определяет временной масштаб работы единичного нейрона
#         t_ltp = 2.0 
#         # Инкремент веса связи
#         alpha_plus = np.random.uniform(80, 120)   
#         # Декремент веса связи
#         alpha_minus = np.random.uniform(40, 60)   
#         # Минимальный вес связи
#         omega_min = 1.0
#         # Максимальный вес связи
#         omega_max = 1200.0

#         # Идем по всем входам
#         for input_id in range(self.input_size):
#             # Время последней активации входа
#             t_input = self.last_input_times[input_id]
#             # Разница во времени, когда отработал нейрон и когда пришел входной спайк
#             delta_t = t_spike - t_input        
#             # Если вход сработал незадолго до спайка, увеличиваем связь
#             if 0 < delta_t < t_ltp:
#                 self.weights[neuron_index][input_id] += alpha_plus
#             else:
#                 # Иначе уменьшаем
#                 self.weights[neuron_index][input_id] -= alpha_minus

#             # Ограничиваем вес
#             self.weights[neuron_index][input_id] = np.clip(
#                 self.weights[neuron_index][input_id],
#                 omega_min,
#                 omega_max
#             )