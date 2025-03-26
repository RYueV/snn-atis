from torchvision import datasets, transforms
import pickle
import numpy
import random


from encoding import mnist_to_aer_events as aer
from core import network as net
import visualization as v



def train(
    snn,
    mnist_dataset,
    images_per_direction=300,
    steps_per_image=12,         #поменять?
    dt=1.5,
    threshold=0.05,
):
    import random

    directions = ["right", "down", "left", "up"]
    total_images = images_per_direction * len(directions)

    all_indices = list(range(total_images))
    random.shuffle(all_indices) #  это странно

    dir_to_indices = {}
    start = 0
    for d in directions:
        dir_to_indices[d] = all_indices[start : (start + images_per_direction)]
        start += images_per_direction

    samples = []
    for d in directions:
        for idx in dir_to_indices[d]:
            samples.append((d, idx))
    random.shuffle(samples)

    # Количество спайков нейронов по направлениям
    neuron_stats = {d: [0 for _ in range(snn.count_neurons)] for d in directions}
    history = []

    # Основной цикл обучения
    for i in range(total_images):
        # Достаем пару (направление сдвига : индекс картинки)
        d, idx = samples[i]
        print(f"{i+1}/{total_images}\tнаправление: {d}, изображение: {idx}")

        # Извлекаем из датасета тензор [1x28x28] и метку
        img_tensor, _ = mnist_dataset[idx]
        # Удаляем первую ось и преобразуем в numpy 2D матрицу
        image = img_tensor.squeeze().numpy()

        # Преобразуем в поток событий
        events = aer.mnist_image_to_events(
            image=image,                # исходное изображение
            steps=steps_per_image,      # количество сдвигов в одном направлении
            direction=d,                # направление сдвига
            dt=dt,                      # время между кадрами
            threshold=threshold         # порог для возникновения события
        )

        # Сброс состояния сети (кроме весов) перед обработкой нового изображения
        snn.reset()
        for t, x, y, p in sorted(events, key=lambda e: e[0]):
            snn.events_to_lif(t, x, y, p)

        # Обновляем статистику спайков
        for _, neuron_id in snn.spikes:
            neuron_stats[d][neuron_id] += 1

        # Вычисляем текущие предпочтения нейронов
        neuron_preference = {}
        for neuron_id in range(snn.count_neurons):
            counts = [neuron_stats[d][neuron_id] for d in directions]
            max_count = max(counts)
            # Если нейрон не выдал ни одного спайка
            if max_count == 0:
                # То предпочитаемое напарвление не определено
                neuron_preference[neuron_id] = None
            else:
                # Ищем индекс направления, которое дает максимальное количество спайков
                neuron_preference[neuron_id] = directions[counts.index(max_count)]

        # Считаем общее количество спайков каждого нейрона
        spike_counts = {i: 0 for i in range(snn.count_neurons)}
        for _, neuron_id in snn.spikes:
            spike_counts[neuron_id] += 1

        # Создаем словарь "голосов" за направление
        direction_votes = {d: 0 for d in directions}
        # Обходим все нейроны
        for neuron_id, count in spike_counts.items():
            # Берем направление, которое предпочитает нейрон
            pref = neuron_preference.get(neuron_id)
            if pref:
                # Прибавляем количество спайков этого нейрона к "голосам" за это неправление
                direction_votes[pref] += count

        # Ищем направление с максимальным количеством голосов
        max_vote = max(direction_votes.values())
        # Берем направления с максимальным количеством голосов
        # (несколько направлений могут иметь одинаковое количество голосов)
        candidates = [d for d in directions if direction_votes[d] == max_vote]
        # Случайный выбор из кандидатов
        predicted = random.choice(candidates)
        # Записываем в историю 1, если предсказание верное
        history.append(1 if predicted == d else 0)

    v.plot_learning(history)

    return neuron_stats, neuron_preference




if __name__ == "__main__":

    # Каждый объект mnist - картинка (матрица) и метка (цифра на изображении)
    mnist = datasets.MNIST(
        root = "./mnist",                   # путь для сохранения
        train = True,                       # обучающая выборка на 60000 картинок
        download = False,        
        transform = transforms.ToTensor()   # [1, 28, 28]; яркость нормирована 0...1
        )
    #print(len(mnist))


    img_tensor, label = mnist[0]
    # #print(label)

    image = img_tensor.squeeze().numpy()
    # image = image*255
    # image = image.astype(numpy.int32)
    # im = Image.fromarray(image)


    

    # events = aer.mnist_image_to_events(
    #     image, 
    #     steps=100,
    #     direction="right",
    #     dt=1.5,
    #     threshold=0.08
    # )
    # aer.plot_event_raster(events, f"Цифра {label}")

    snn = net.SNNnetwork(
            count_neurons=24,
            alpha_plus=40,
            alpha_minus=20,
            beta_plus=2.0,
            beta_minus=2.0,
            t_inhibit=1.5
    )
    #dir_stats, neuron_stats = train(snn, mnist)


    #v.plot_total_spikes_per_direction(stats)
    #v.plot_neuron_activity(stats)

    #analyze_directions(snn, mnist, num_imgs=20)

    train_stats, neuron_pref = train(snn, mnist, images_per_direction=500)

    v.plot_neuron_stats(train_stats)

    print()
    print(neuron_pref)

    # Сохраняем веса
    with open("snn_weights.pkl", "wb") as f:
        pickle.dump(snn.weights, f)

    
    # with open("snn_weights.pkl", "rb") as f:
    #     loaded_weights = pickle.load(f)

    # snn.weights = loaded_weights

    # test_direction_prediction(
    #     snn=snn,
    #     mnist=mnist,
    #     neuron_preference=neuron_stats,
    #     num_imgs=10,
    #     print_details=True
    # )

    # for t, x, y, p in sorted(events, key=lambda e: e[0]):
    #     snn.events_to_lif(t, x, y, p)

    # # Растр спайков
    # v.spike_raster(snn.spikes)

    # for i in range(4): 
    #     v.show_neuron_weights(snn.weights, neuron_index=i)



