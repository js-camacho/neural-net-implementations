"""
Created on Mon Feb 17 19:32:14 2020

@author: Michael Nielsen
network.py
~~~~~~~~~~

A module to implement the stochastic gradient descent learning
algorithm for a feedforward neural network.  Gradients are calculated
using backpropagation.  Note that I have focused on making the code
simple, easily readable, and easily modifiable.  It is not optimized,
and omits many desirable features.
"""

#%% Importing Libraries

# Standard library
import random
from datetime import datetime

# Third-party libraries
import numpy as np
import pandas as pd

#%% Neural Network Implementation

class Network(object):

    def __init__(self, sizes):
        """The list <<sizes>> contains the number of neurons in the respective layers of the network.  
        For example, if the list was [2, 3, 1] then it would be a three-layer network, with the first layer 
        containing 2 neurons, the second layer 3 neurons, and the third layer 1 neuron.  
        
        The biases and weights for the network are initialized randomly, using a Gaussian (Normal) distribution 
        with mean 0, and variance 1.  
        
        Note that the first layer is assumed to be an input layer, and by  convention we won't set any biases 
        for those neurons, since biases are only ever used in computing the outputs from later layers."""
        
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """
        Return the output of the next layer of the network if <<a>> are the activations of a previous layer.
        It is important that the input is a (n, 1) numpy array, not a (n, ) vector:
            input:  [0, 1]              Error
                    np.array([0,1])     Error
                    np.array([[0],[1]]) OK
        """
        
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        """Train the neural network using mini-batch Stochastic Gradient Descent.  
        The <<training_data>> is a list of tuples (x, y) representing the training inputs (x) and the desired 
        outputs (y). The <<method>> must be either V for vectorial backpropagation or 'M' for matrix 
        backpropagation. <<cost_fun>> defines the cost function, can be CE for cross-entropy or Q for quadratic
        
        The other non-optional parameters are self-explanatory.  
        
        If <<test_data>> is provided then the network will be evaluated against the test data after each epoch, 
        and partial progress printed out.  This is useful for tracking progress, but slows things down 
        substantially."""
        
        if test_data: 
            n_test = len(test_data)
        
        n = len(training_data)
        start = datetime.now()      # Starting timer 
        
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            
            # Backpropagation
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)  
                    
            stop = datetime.now() - start  # Stopping timer 
            if test_data:   
                stop = datetime.now() - start  # Stopping timer
                print ('Epoch ',j,': ',self.evaluate(test_data),' / ', n_test,'\t  Time elapsed:',stop)
            else:
                print ('Epoch',j,'complete\tTime elapsed:',stop)
               
        stop = datetime.now() - start  # Stopping timer        
        print('Total time elpased:', stop)

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying gradient descent using backpropagation to a single 
        mini batch.
        
        The <<mini_batch>> is a list of tuples (x, y), and <<eta>> is the learning rate."""
        
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w-(eta/len(mini_batch))*nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y, cost_fun = 'CE'):
        """Return a tuple <<(nabla_b, nabla_w)>> representing the gradient for the cost function C_x. 
        
        <<cost_fun>> can be CE for cross-entropy or Q for quadratic
        
        <<nabla_b>> and <<nabla_w>> are layer-by-layer lists of numpy arrays, similar to <<self.biases>> and 
        <<self.weights>>."""
        
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # Feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # Backward pass
        if cost_fun == 'CE':
            delta = self.cost_derivative(activations[-1], y)                            # (BP1)
        elif cost_fun == 'Q':
            delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])    # (BP1)
        nabla_b[-1] = delta                                                         # (BP3)
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())                    # (BP4)
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp              # (BP2)
            nabla_b[-l] = delta                                                     # (BP3)
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())              # (BP4)
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, a, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (a-y)

# Squishification function and its derivate
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

#%% Importing training and test data
 
# Training data (60.000 images of 28x28 pixels)
mnist_train = pd.read_csv('mnist_train.csv', header = 0)
training_data = []
n_inputs = mnist_train.shape[1] - 1

for index, row in mnist_train.iterrows():
    reshaped = np.array(row[1:]).reshape((n_inputs, 1))     # Changing from (n,) to (n, 1) array
    reshaped = reshaped/256                                 # Converting to a [0, 1] range
    y_vector = np.zeros((10,1), dtype = int)                # Creating a (m, 1) array to store the desired outputs
    y_vector[row[0]] = 1                                    # For example: y = 2 -> y = [0 0 1 ... 0]
    training_data.append((reshaped, y_vector))
    
# Test data (10.000 images of 28x28 pixels)

mnist_test = pd.read_csv('mnist_test.csv', header = 0)
testing_data = []

for index, row in mnist_test.iterrows():
    reshaped = np.array(row[1:]).reshape((n_inputs, 1))     # Changing from (n,) to (n, 1) array
    reshaped = reshaped/256                                 # Converting to a [0, 1] range
    testing_data.append((reshaped, row[0]))   
    
del index, row, reshaped

#%% Experiments

net = Network([784, 300, 10])
net.SGD(training_data, epochs = 10, mini_batch_size = 10 , eta = 0.5, test_data = testing_data)

#%% Example printer

row = random.randint(0,10000)
for i in range(28):
    for j in range(28):
        if mnist_test.iloc[row, j + 1 + 28*i] == 0:
            response = '-'
        else:
            response = '#'
        print(response, end = ' ')
    print('')
print('********************')    
print('Row:',row,'\nNetwork Guess:',np.argmax(net.feedforward(testing_data[row][0])),'\nReal Value:', testing_data[row][1])
print('********************')
