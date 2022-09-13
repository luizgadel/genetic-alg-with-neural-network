import numpy as np


class Network:
    def __init__(self, input_layer_size, hidden_layer_size, output_layer_size, weights, biases):
        self.il_size = input_layer_size
        self.hl_size = hidden_layer_size
        self.ol_size = output_layer_size
        self.biases = biases  # shape = [1x7, 1x5]
        self.weights = weights  # shape = [2x7, 7x5]
        self.num_layers = 3

    def feed_forward(self, x):
        activation = x
        self.activations = [x]
        self.zs = []

        for w, b in zip(self.weights, self.biases):
            z = np.dot(activation, w) + b
            self.zs.append(z)
            activation = self.sigmoid(z)
            self.activations.append(activation)

        return activation

    def sigmoid(self, s):
        return 1/(1 + np.exp(-s))

    def sigmoid_prime(self, s):
        return self.sigmoid(s) * (1 - self.sigmoid(s))

    def backprop(self, x, y):
        """Return a tuple "(nabla_b, nabla_w)" representing the
        gradient for the cost function C_x.  "nabla_b" and
        "nabla_w" are layer-by-layer lists of numpy arrays, similar
        to "self.biases" and "self.weights"."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # feedforward
        self.outputs = []
        activation = np.array(x, ndmin=2)
        # list to store all the activations, layer by layer
        self.activations = []
        self.activations.append(activation)
        zs = []  # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(activation, w) + b
            zs.append(z)
            activation = self.sigmoid(z)
            self.activations.append(activation)

        self.outputs.append(activation)

        # backward pass
        delta = self.cost_derivative(self.activations[-1], y) * \
            self.sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(self.activations[-2].T, delta)
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = self.sigmoid_prime(z)  # (1x7)
            delta = np.dot(delta, self.weights[-l+1].T) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(self.activations[-l-1].T, delta)
        return (nabla_b, nabla_w)

    def train(self, X, Y):
        self.X = X
        self.Y = Y
        self.outputs = []
        for x, y in zip(X, Y):
            output = self.feed_forward(x)
            self.outputs.append(output)
            # backprop(x, y)

    def update_mini_batch(self, X, Y, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The "mini_batch" is a list of tuples "(x, y)", and "eta"
        is the learning rate."""
        self.X = X
        self.Y = Y
        self.batch_size = len(X)

        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in zip(X, Y):
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights = [w-(eta/self.batch_size)*nw for w,
                        nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/self.batch_size)*nb for b,
                       nb in zip(self.biases, nabla_b)]

    def cost(self):
        square_error = np.square(self.Y - self.activations[-1])
        return np.mean(square_error), square_error

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)
