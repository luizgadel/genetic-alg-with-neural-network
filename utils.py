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