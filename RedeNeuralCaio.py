import numpy as np

class NNC(object):
    def __init__(self):
        #parameters
        self.inputSize = 2
        self.outputSize = 5
        self.hiddenSize = 7

        #weights
        self.W1 = np.random.randn(self.inputSize, self.hiddenSize) #(7x2) weight matrix between input and hidden Layer
        self.W2 = np.random.randn(self.hiddenSize, self.outputSize) #(5x7) weight matrix between hidden and output Layer
    
    def feed_forward(self, X):
        #foward propagation through the network
        self.z = np.dot(X, self.W1) #dot product of X (input) and first set of weights (7x2)
        self.z2 = self.sigmoid(self.z) #activation function
        self.z3 = np.dot(self.z2, self.W2) #dot product of hidden Layer (z2) and second set of weights (5x7)
        output = self.sigmoid(self.z3)
        return output
    
    def sigmoid(self, s, deriv=False):
        if (deriv == True):
            return s * (1 - s)
        return 1/(1 + np.exp(-s))

    def backward(self, X, Y, output):
        #backward propagate through the network
        self.output_error = Y - output #error in output
        self.output_delta = self.output_error * self.sigmoid(output, deriv = True)

        self.z2_error = self.output_delta.dot(self.W2.T) #z2 error: how much our hidden layer weights contribute to output error
        self.z2_delta = self.z2_error * self.sigmoid(self.z2, deriv=True) #applying derivative of sigmoid to z2 error

        self.W1 += X.T.dot(self.z2_delta) #adjusting first set (input -> hidden) weights
        self.W2 += self.z2.T.dot(self.output_delta) #adjusting second set (hidden -> output) weights
    
    def train(self, X, Y):
        output = self.feed_forward(X)
        self.backward(X, Y, output)