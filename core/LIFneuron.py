import numpy as np

# Модель LIF; изменение потенциала нейрона только при событии
class LIF_neuron_event:
    def __init__(
            self, 
            tau_leak = 12.4319,     # постоянная времени утечки (мс) 
            I_thres = 1792.6045,    # пороговое значение потенциала для генерации спайка
            t_ref = 2.7983,         # длительность периода после спайка, когда нейрон не реагирует (мс)
            neuron_id=None          # id нейрона 
    ):
        """

        Потенциал обновляется только при входных событиях;
        При изменении потенциала используется экспоненциальное затухание:
        u = u * exp(-(t_spike - t_last_spike)/tau_leak) + w,
            w - вес связи со входом
            t_spike - время текущего входа
            t_last_spike - время последнего входного воздействия

        Если u > I_thres, генерируется спайк.

        После спайка u обнуляется и нейрон не реагирует на воздействие t_ref мс


        """
        # Параметры модели
        self.tau_leak = tau_leak
        self.I_thres = I_thres
        self.t_ref = t_ref     
        self.id = neuron_id
        
        # Переменные состояния нейрона
        self.u = 0.0                        # уровень входного сигнала
        self.last_input_time = 0.0          # время последнего входного воздействия
        self.last_spike_time = -np.inf      # время последнего спайка
        self.spikes = []                    # моменты спайков
        self.inhibited_until = 0.0          # время окончания ингибирования


    # Сброс нейрона к начальному состоянию
    def reset(self):
        self.u = 0.0
        self.last_input_time = 0.0
        self.last_spike_time = -np.inf
        self.spikes = []
        self.inhibited_until = 0.0


    # Обработка входного воздействия
    def process_input(
            self,
            input_time,
            weight
    ):
        spiked = False

        # Если не вышло время ингибирования или не закончился рефрактерный период, 
        # то нейрон не реагирует на внешнее воздействие
        if (
            input_time < self.inhibited_until or
            input_time < self.last_spike_time + self.t_ref
        ):
            return spiked, self.u

        # Иначе - экспоненциальное затухание сигнала
        delta_t = input_time - self.last_input_time
        self.u *= np.exp(- delta_t / self.tau_leak)
        self.u += weight
        # Обновляем время последнего входного воздействия
        self.last_input_time = input_time

        # Проверяем порог
        if self.u > self.I_thres:
            spiked = True
            self.spikes.append(input_time)
            self.last_spike_time = input_time
            self.u = 0
            
        return spiked, self.u

