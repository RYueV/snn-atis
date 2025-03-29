from torchvision import datasets, transforms
import random
import numpy as np


from .operators import (
    generate_params,
    mix_params,
    mutate_params,
    candidate_selection
)
from .train_snn import evaluate_snn_accuracy



# Генетический алгоритм для подбора оптимальных параметров SNN
def genetic_search(
        mnist_dataset,              # загруженный датасет mnist
        pop_size=10,                # размер популяции 
        generations=5,              # количество поколений
        images_per_direction=50,    # кол-во изображений на каждое направление при запуске теста
        epochs=1,                   # кол-во эпох обучения для каждой особи
        best_rate=0.2,              # доля лучших особей, которые сохраняются без изменений
        parents_rate=0.5,           # доля выживших для отбора родителей
        mutation_prob=0.3           # вероятность мутации каждого параметра
):
    """

    Возвращает словарь лучших параметров за все время и точность предсказаний при них.

    """

    # Инициализируем первую популяцию
    population = [generate_params() for _ in range(pop_size)]
    # Лучшая точность за все время
    best_global_accuracy = -1.0
    # Параметры SNN, дающие эту точность
    best_global_params = None


    # Генерируем заданное количество поколений
    for generation in range(generations):
        print(f"\n\nПоколение {generation+1}/{generations}")

        # Храним статистику точности для всего поколения
        population_accuracy = []
        # Обходим пары (номер индивида, его параметры)
        for i, indiv_params in enumerate(population):
            print(f"Индивид {i+1} / {pop_size}")

            # Запускаем обучение SNN
            accuracy = evaluate_snn_accuracy(
                params=indiv_params,                        # параметры SNN для обучения
                mnist_dataset=mnist_dataset,                # загруженный датасет
                images_per_direction=images_per_direction,  # кол-во изображений на одно направление
                epochs=epochs                               # сколько раз повторяем обучение на фикс. выборке
            )

            # Сохраняем точность
            population_accuracy.append((accuracy, indiv_params))
            print(f"Точность = {accuracy*100:.2f}%")
            

        # Сортируем популяцию по точности (от лучшего к худшему)
        population_accuracy.sort(key=lambda x: x[0], reverse=True)
        # Выбираем лучшую особь
        best_accuracy, best_params = population_accuracy[0]

        print(f"\n\nЛучшая особь в поколении: {best_accuracy*100:.2f}%")
        print(f"Параметры:\n{best_params}")

        # Сохраняем лучшие за поколение параметры, если они лучшие за все время
        if best_accuracy > best_global_accuracy:
            best_global_accuracy = best_accuracy
            best_global_params = best_params


        # Сохраняем указанное количество лучших особей, 
        # они перейдут в следующее поколение без изменений
        best_count = int(pop_size * best_rate)
        best_indiv = population_accuracy[:best_count]

        # Отбираем родителей
        # Количество родителей должно быть не меньше best_count для честного отбора
        num_parents = max(best_count, int(pop_size * parents_rate))
        parents = population_accuracy[:num_parents]

        # Формируем следующее поколение, начинаем с лучших особей
        next_generation = [params for (_, params) in best_indiv]

        # Создаем потомков, пока не дойдем до заданного размера популяции
        while len(next_generation) < pop_size:
            # Отбираем двух родителей
            _, parent1 = candidate_selection(parents, group_size=2)
            _, parent2 = candidate_selection(parents, group_size=2)

            # Формируем параметры потомка из родительских
            child = mix_params(parent1, parent2)
            # Изменяем часть параметров путем мутаций
            child = mutate_params(child, mutation_prob=mutation_prob)

            # Добавляем потомка в популяцию
            next_generation.append(child)

        # Переходим к следующему поколению
        population = next_generation

    # Возвращаем лучшие параметры за все время и точность, которую они дают
    return best_global_params, best_global_accuracy




# Запуск генетического алгоритма
if __name__ == '__main__':
    # Фиксируем random
    random.seed(42)
    np.random.seed(42)

    # Загружаем mnist
    mnist = datasets.MNIST(
        root='data/mnist',
        train=True,
        download=False,
        transform=transforms.ToTensor()
    )

    # Запускаем генетический алгоритм
    best_params, best_accuracy = genetic_search(
        mnist,                      # загруженный датасет mnist
        pop_size=10,                # размер популяции 
        generations=5,              # количество поколений
        images_per_direction=50,    # кол-во изображений на каждое направление при запуске теста
        epochs=1,                   # кол-во эпох обучения для каждой особи
        best_rate=0.2,              # доля лучших особей, которые сохраняются без изменений
        parents_rate=0.5,           # доля выживших для отбора родителей
        mutation_prob=0.3           # вероятность мутации каждого параметра
    )

    print(f"\n\n\nЛучшие параметры за все время:\n{best_params}")
    print(f"Точность: {best_accuracy*100:.2f}%")
