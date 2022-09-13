import numpy as np
import time
from sklearn.model_selection import train_test_split
from network import Network
import matplotlib.pyplot as plt


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
    while (i < size):
        if (array[i] < min_value):
            return True
        else:
            i += 1

    if (i == size):
        return False


X = np.array(([0, 0], [0, 0.27], [0, 0.71],
              [0.3, 0.27], [0.3, 0.71], [0.3, 1],
              [0.5, 0], [0.5, 0.27], [0.5, 1],
              [0.75, 0.27], [0.75, 0.71], [0.75, 1],
              [1, 0], [1, 0.71], [1, 1]), dtype=float)

Y = np.array(([1, 1, 0, 0.85, 0.82],
              [0.78, 0.85, 0.04, 0.9, 0.9],
              [0.75, 0.8, 0.08, 0.97, 0.96],
              [0.7, 0.72, 0.29, 0.82, 0.63],
              [0.45, 0.49, 0.3, 0.91, 0.71],
              [0.42, 0.46, 0.32, 0.94, 0.82],
              [0.64, 0.72, 0.32, 0.74, 0.51],
              [0.42, 0.49, 0.4, 0.8, 0.28],
              [0.33, 0.33, 0.56, 0.85, 0.75],
              [0.31, 0.26, 0.49, 0.5, 0.28],
              [0.28, 0.23, 0.53, 0.63, 0.37],
              [0.22, 0.15, 0.65, 0.69, 0.49],
              [0.11, 0.28, 0.54, 0, 0],
              [0.03, 0.13, 0.9, 0.22, 0.28],
              [0, 0, 1, 0.35, 0.35]), dtype=float)

train_X, test_X = train_test_split(X, test_size=0.25)
train_Y, test_Y = train_test_split(Y, test_size=0.25)

def train_neural_network(nn: Network, n_gen):
    starting_time = time.time()

    x = range(n_gen)
    plot_x = range(0, n_gen, 10)
    plot_train_y = []
    plot_test_y = []
    for i in x:
        nn.update_mini_batch(train_X, train_Y, 0.5)

        if (i % 10 == 0):
            train_cost, train_square_error = nn.cost()
            plot_train_y.append(train_cost)

            nn.train(test_X, test_Y)
            test_cost, test_square_error = nn.cost()
            plot_test_y.append(test_cost)

    ending_time = time.time()
    elapsed_time = ending_time - starting_time

    print("Fim do treino da rede neural.\n")
    print(f"Número total de gerações: {n_gen}")
    print('Tempo de execução:', round(elapsed_time, 3), 'segundos.')
    print(train_square_error)
    print(test_square_error)
    plot_MSE(plot_x, plot_train_y, plot_test_y)


def train_genetic_algorithm(ag, n_gen):
    starting_time = time.time()

    plot_x = range(0, n_gen, 10)
    plot_train_y = []
    plot_test_y = []
    for i in range(n_gen):
        ag.new_generation()

        if (i % 10 == 0):
            nn: Network = ag.last_gen_best_nn
            train_cost = nn.cost()
            plot_train_y.append(train_cost)

            nn.train(test_X, test_Y)
            test_cost = nn.cost()
            plot_test_y.append(test_cost)

    ending_time = time.time()
    elapsed_time = ending_time - starting_time

    print("Fim do treino do algoritmo genético.\n")
    print(f"Número total de gerações: {n_gen}")
    print('Tempo de execução:', round(elapsed_time, 3), 'segundos.')
    plot_MSE(plot_x, plot_train_y, plot_test_y)


def split_weights_and_biases(weights_and_biases, l1_size, l2_size):
    w_size = l1_size * l2_size
    weights = weights_and_biases[:w_size].reshape(l1_size, l2_size)
    biases = weights_and_biases[w_size:]
    return weights, biases


def plot_MSE(x, y_train, y_test):
    train_cost = y_train[-1]
    print("MSE-train: " + str(train_cost))

    test_cost = y_test[-1]
    print("MSE-test:  " + str(test_cost))

    plt.plot(x, y_train, 'b:')
    plt.plot(x, y_test, 'r:')
    plt.xlabel("Gerações")
    plt.ylabel("Erro Quadrático Médio")
    plt.title("MSE de treino e teste de uma RN")
    plt.legend(["treino", "teste"])
    plt.show()
