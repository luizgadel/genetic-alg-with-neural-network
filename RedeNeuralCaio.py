import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, weights_1, weights_2):
        #parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        '''
        Os pesos precisarão ser definidos através do construtor.
        '''
        #weights
        self.W1 = weights_1
        self.W2 = weights_2
        
    def sigmoid(self, s):
        return 1/(1 + np.exp(-s))

    def sigmoid_prime(self, s):
        return self.sigmoid(s) * (1 - self.sigmoid(s))
        
    def feed_forward(self, X, Y):
        self.X = X
        self.Y = Y
        #foward propagation through the network
        dot_product_A_1 = np.dot(X, self.W1) #dot product of X (input) and first set of weights (7x2)
        self.A_1 = self.sigmoid(dot_product_A_1) #activation function
        
        dot_product_A_2 = np.dot(self.A_1, self.W2) #dot product of hidden Layer (A_1) and second set of weights (5x7)
        self.A_2 = self.sigmoid(dot_product_A_2)
        return self.A_2
    
    def backward_propagation(self):
        #backward propagate through the network
        self.ol_error = self.Y - self.A_2 #error in output
        self.delta_2 = self.ol_error * self.sigmoid_prime(self.A_2)
        
        self.hl_error = np.dot(self.delta_2, self.W2.T) #hl error: how much our hidden layer weights contribute to output error
        self.delta_1 = self.hl_error * self.sigmoid_prime(self.A_1) #applying derivative of sigmoid to A_1 error
        
        self.W1 += np.dot(self.X.T, self.delta_1) #adjusting first set (input -> hidden) weights
        self.W2 += np.dot(self.A_1.T, self.delta_2) #adjusting second set (hidden -> output) weights

    def cost(self):
        square_error = np.square(self.Y - self.A_2)
        return np.mean(square_error), square_error

    