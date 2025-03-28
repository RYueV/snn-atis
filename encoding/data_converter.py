import pickle
import numpy as np
from PIL import Image


# Преобразует тензон pytorch в изображение
def tensor_to_image(image_tensor, save_path=None):
    image = image_tensor.squeeze().numpy()
    image = image * 255.0
    image = image.astype(np.uint8)
    img = Image.fromarray(image)
    img.show()
    if save_path is not None:
        img.save(save_path)



# Сохраняет веса в бинарный файл
def list_to_binary(filename, data):
    """
    
    weights: матрица весов объекта snn

    """
    with open(filename, "wb") as f:
        pickle.dump(data, f)



# Извлекает веса из бинарного файла
def binary_to_list(filename="data/snn_weights.pkl"):
    """
    
    filename: полный путь к файлу с весами

    """
    with open(filename, "rb") as f:
        loaded_data = pickle.load(f)

    return loaded_data