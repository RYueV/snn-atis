import numpy as np

# Модель LIF; изменение потенциала нейрона только при событии
class LIF_neuron_event:
    def __init__(
            self, 
            tau_leak = 10.0,     # определяет, как быстро потенциал стремится к покою 
            I_thres = 10000,    # определяет селективность нейрона (макс. значение на входе)
            t_ref = 15.0,       # длительность периода после спайка, когда нейрон не реагирует 
            neuron_id=None      # id нейрона для отладки сети
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





# # Стандартная модель LIF
# class LIF_neuron:
#     def __init__(
#             self, 
#             tau_m = 5.0,    # определяет, как быстро потенциал стремится к покою 
#             t_ref = 10.0,   # длительность периода после спайка, когда нейрон не реагирует 
#             dt = 1.0,       # шаг интегрирования; определяет временной шаг симуляции
#             V_thresh = 1.0, # пороговый потенциал, при котором генерируется спайк
#             V_reset = 0.0,  # потенциал сброса после спайка; обычно V_reset = V_rest 
#             V_rest = 0.0,   # потенциал покоя
#             R = 5.0,        # коэффициент, регулирующий вклад входного тока
#             neuron_id=None  # id нейрона для отладки сети
#     ):
#         """

#         Уравнение модели:
#         V(t + dt) = V(t) + dt/tau_m * (-(V - V_rest) + RI(t))

#         """
#         self.id = neuron_id

#         # Параметры модели
#         self.tau = tau_m
#         self.t_ref = t_ref
#         self.dt = dt
#         self.V_thresh = V_thresh  
#         self.V_reset = V_reset  
#         self.V_rest = V_rest    
#         self.R = R           
        
#         # Переменные состояния нейрона
#         self.V = self.V_rest        # текущий мембранный потенциал
#         self.time = 0.0             # текущее время симуляции
#         self.ref_time_left = 0      # оставшееся время в рефрактерном периоде
#         self.spikes = []        # моменты, когда происходили спайки
#         self.inhibited_until = 0.0


#     # Сброс нейрона к начальному состоянию
#     def reset(self):
#         self.V = self.V_rest
#         self.ref_time_left = 0
#         self.time = 0.0
#         self.spikes = []
#         self.inhibited_until = 0.0


#     # Один шаг симуляции
#     def step(self, input_signal):
#         """

#         input_signal - величина входного сигнала
#         Если на шаге происходит спайк, функция возвращает True

#         """
#         spiked = False

#         # Время ингибирования складывается с периодом рефрактерности
#         if self.time < self.inhibited_until:
#             self.time += self.dt
#             return spiked, self.V
#         # Если в рефрактерном периоде, то не реагируем на входной ток
#         if self.ref_time_left > 0:
#             self.ref_time_left -= 1
#         else:   # Иначе считаем разность потенциалов 
#             dV = (-(self.V - self.V_rest) + self.R * input_signal) / self.tau
#             # Обновляем потенциал с учетом величины шага
#             self.V += dV * self.dt

#         # Проверяем порог
#         if self.V >= self.V_thresh:
#             # Если больше порогового, то регистрируем время импульса
#             spiked = True
#             self.spikes.append(self.time)   
#             # Сбрасываем потенциал
#             self.V = self.V_reset
#             # Устанавливаем рефрактерный период, если модель требует
#             if self.t_ref > 0:
#                 # Значение высчитывается в количестве шагов step
#                 self.ref_time_left = int(self.t_ref / self.dt)

#         # Увеличиваем модельное время
#         self.time += self.dt
#         return spiked, self.V
    



