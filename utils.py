import numpy as np

def get_roulette_wheel(array):
    normalized_arr = array / np.sum(array)
    arr_size = len(array)

    roulette_wheel = []
    anterior = 0
    for i in range(arr_size):
        atual = round(normalized_arr[i]*10000000)
        nova_pos = anterior + atual
        roulette_wheel.append(nova_pos)

        anterior = nova_pos
    
    return roulette_wheel

def random_2D_list(dimension_1_size, dimension_2_size):
    return np.random.randn(dimension_1_size, dimension_2_size)

def all_items_below_90_percent(array, max_value):
    min_value = max_value*0.9
    size = len(array)
    i = 0
    while(i < size):
        if (array[i] < min_value):
            return True
        else: 
            i += 1

    if (i == size):
        return False

train_X = np.array(([0, 0], [0, 0.27], [0, 0.71],
              [0.3, 0.27], [0.3, 0.71], [0.3, 1],
              [0.5, 0], [0.5, 0.27], [0.5, 1],
              [0.75, 0.27], [0.75, 0.71], [0.75, 1],
              [1, 0], [1, 0.71], [1, 1]), dtype=float)

train_Y = np.array(([1, 1, 0, 0.85, 0.82], [0.78, 0.85, 0.04, 0.9, 0.9], [0.75, 0.8, 0.08, 0.97, 0.96],
              [0.7, 0.72, 0.29, 0.82, 0.63], [0.45, 0.49, 0.3,
                                              0.91, 0.71], [0.42, 0.46, 0.32, 0.94, 0.82],
              [0.64, 0.72, 0.32, 0.74, 0.51], [0.42, 0.49, 0.4,
                                               0.8, 0.28], [0.33, 0.33, 0.56, 0.85, 0.75],
              [0.31, 0.26, 0.49, 0.5, 0.28], [0.28, 0.23, 0.53,
                                              0.63, 0.37], [0.22, 0.15, 0.65, 0.69, 0.49],
              [0.11, 0.28, 0.54, 0, 0], [0.03, 0.13, 0.9, 0.22, 0.28], [0, 0, 1, 0.35, 0.35]), dtype=float)