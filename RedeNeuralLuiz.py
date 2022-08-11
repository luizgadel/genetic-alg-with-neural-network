import numpy as np

def random_weight_list(input_size, output_size):
    return np.random.randn(input_size, output_size)

class NNL:
    def __init__(self, input_size, hidden_size, output_size):
        #parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        #weights
        self.W1 = random_weight_list(input_size, hidden_size)
        self.W2 = random_weight_list(hidden_size, output_size)
        
    def sigmoid(self, s, deriv = False):
        if (deriv == True):
            return s * (1 - s)
        else:
            return 1/(1 + np.exp(-s))
        
    def feed_forward(self, X):
        #foward propagation through the network
        dot_product_A_1 = np.dot(X, self.W1) #dot product of X (input) and first set of weights (7x2)
        self.A_1 = self.sigmoid(dot_product_A_1) #activation function
        
        dot_product_A_2 = np.dot(self.A_1, self.W2) #dot product of hidden Layer (A_1) and second set of weights (5x7)
        self.A_2 = self.sigmoid(dot_product_A_2)
        return self.A_2
    
    def backward_propagation(self, X, Y):
        #backward propagate through the network
        self.ol_error = Y - self.A_2 #error in output
        self.delta_2 = self.ol_error * self.sigmoid(self.A_2, deriv = True)
        
        self.hl_error = np.dot(self.delta_2, self.W2.T) #hl error: how much our hidden layer weights contribute to output error
        self.delta_1 = self.hl_error * self.sigmoid(self.A_1, deriv = True) #applying derivative of sigmoid to A_1 error
        
        self.W1 += np.dot(X.T, self.delta_1) #adjusting first set (input -> hidden) weights
        self.W2 += np.dot(self.A_1.T, self.delta_2) #adjusting second set (hidden -> output) weights
        
    def train(self, X, Y):
        self.feed_forward(X)
        self.backward_propagation(X, Y)