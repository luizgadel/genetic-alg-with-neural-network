import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, weights_biases_1, weights_biases_2):
        #parameters
        self.input_size = input_size #2
        self.hidden_size = hidden_size #7
        self.output_size = output_size #5
        
        '''
        Os pesos precisarão ser definidos através do construtor.
        '''
        #weights
        self.W1 = weights_biases_1 #(3x7)
        self.W2 = weights_biases_2 #(8x5)
        
    def sigmoid(self, s, deriv = False):
        if (deriv == True):
            return s * (1 - s)
        else:
            return 1/(1 + np.exp(-s))
        
    def feed_forward(self, X):
        #foward propagation through the network
        train_size = X.shape[0]
        X_extended = np.column_stack((X, np.ones(train_size))) #adiciono uma coluna de 1s para soma do bias no produto de matrizes. Res.: (Tx3)
        dot_product_A_1 = np.dot(X_extended, self.W1) #dot product of X (input, Tx3) and first set of weights (3x7). Res.: (Tx7)
        self.A_1 = self.sigmoid(dot_product_A_1) #activation function, (Tx7)
        
        A_1_extended = np.column_stack((self.A_1, np.ones(train_size))) #adiciono uma coluna de 1s para soma do bias no produto de matrizes. Res.: (Tx8)
        dot_product_A_2 = np.dot(A_1_extended, self.W2) #dot product of hidden Layer (A_1, Tx8) and second set of weights (8x5). Res.: (Tx5)
        self.A_2 = self.sigmoid(dot_product_A_2) #(Tx5)
        return self.A_2
    
    def backward_propagation(self, X, Y):
        #backward propagate through the network
        self.ol_error = Y - self.A_2 #error in output (Tx5)
        self.delta_2 = self.ol_error * self.sigmoid(self.A_2, deriv = True) #(Tx5)
        
        self.hl_error = np.dot(self.delta_2, self.W2.T) #hl error: how much our hidden layer weights contribute to output error. Res.: (Tx8)
        self.delta_1 = self.hl_error * self.sigmoid(self.A_1, deriv = True) #applying derivative of sigmoid to A_1 error
        
        self.W1 += np.dot(X.T, self.delta_1) #adjusting first set (input -> hidden) weights
        self.W2 += np.dot(self.A_1.T, self.delta_2) #adjusting second set (hidden -> output) weights
        
    def train(self, X, Y):
        self.feed_forward(X)
        self.backward_propagation(X, Y)