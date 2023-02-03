"""
Created on Mon May  4 16:28:53 2020
@author: johan camacho
"""
#%%**************************************************
# GENERAL NEURAL NETWORK IMPLEMENTATION
#****************************************************
'''
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
TOPICS:
-
    *
        +
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
************************************************************
TIPS:
-
************************************************************
'''
#%% Importing Libraries

# Standard library
import random
from datetime import datetime

# Third-party libraries
import numpy as np
import pandas as pd
import PIL as pil
import matplotlib.pyplot as plt

#%% Importing training and test data
print('Reading MNIST data -------------------------------')
start = datetime.now()      # Starting timer
# Training data (60.000 images of 28x28 pixels)
mnist_train = pd.read_csv('mnist_train.csv', header = 0)
training_data = []
training_data2 = []

n_inputs = mnist_train.shape[1] - 1

for index, row in mnist_train.iterrows():
    reshaped = np.array(row[1:]).reshape((n_inputs, 1))     # Changing from (n,) to (n, 1) array
    reshaped = reshaped/256                                 # Converting to a [0, 1] range
    training_data2.append((reshaped, row[0]))               # This set stores <<y>> as a label not a vector, used to evaluate accuracy
    y_vector = np.zeros((10,1), dtype = int)                # Creating a (m, 1) array to store the desired outputs
    y_vector[row[0]] = 1                                    # For example: y = 2 -> y = [0 0 1 ... 0]
    training_data.append((reshaped, y_vector))
print('Training inputs =',len(training_data))

# Test data (10.000 images of 28x28 pixels)

mnist_test = pd.read_csv('mnist_test.csv', header = 0)
testing_data = []

for index, row in mnist_test.iterrows():
    reshaped = np.array(row[1:]).reshape((n_inputs, 1))     # Changing from (n,) to (n, 1) array
    reshaped = reshaped/256                                 # Converting to a [0, 1] range
    testing_data.append((reshaped, row[0]))   
print('Test inputs =',len(testing_data))
    
del index, row, reshaped

stop = datetime.now() - start  # Stopping timer        
print('Total time elpased:', stop)
print('/Reading MNIST data ------------------------------')

#%% Example printer
def print_example(net, row = random.randint(0,10000), failed = False):
    if failed:
        while np.argmax(net.feedforward(testing_data[row][0])) == testing_data[row][1]:
            row = random.randint(0,10000)
    mat = []
    for i in range(28):
        mat_row = []
        for j in range(28):
            mat_row.append(mnist_test.iloc[row, j + 1 + 28*i])            
        '''
            if mnist_test.iloc[row, j + 1 + 28*i] == 0:
                response = '-'
            else:
                response = '#'
            print(response, end = ' ')
            
        print('')
        '''
        mat.append(mat_row)
    mat = np.array(mat)
    print('********************')    
    print('Row:',row,'\nNetwork Guess:',np.argmax(net.feedforward(testing_data[row][0])),'\nReal Value:', testing_data[row][1])
    print('********************')
    
    plt.matshow(mat)
    plt.show()

#%% Network class

class Network(object):
    
    def __init__(self, *layers, activation_function = 'Sigmoid', cost_function = 'Cross-Entropy', 
                 regularize = 0):
        '''
        Parameters
        ----------
        *layers : tuple
            DESCRIPTION: Determine the characteristics of each layer of the Network
            SYNTAX: (<type>, <number_of_neurons>, <extra_arguments>)
                - <type>: str
                    ~ 'Full': Fully-connected layer, first and last layer must be this type
                    ~ 'Conv': Convolutional layer, requires <extra_arguments>
                - <number_of_neurons>: int
                    Number of neurons in the layer
                - <extra_arguments>: tuple of string
                    ~ For 'Conv': (<x>, <y>, <stride>, <maps>, <pool_x>, <pool_y>) : (int, int, int, int, int, int)
                        <x> and <y> are the local receptive field length and heigth, respectively
                        <stride> is the number of neurons that separate each local receptive field
                        <maps> is the number of feature maps in that convolutional layer
                        <pool_x> and <pool_y> are the max_pool size length and heigth, respectively
        
        activation_function : str, optional
            DESCRIPTION: Type of activation function that will be used to feedforward the Network
            VALUES:
                - 'Sigmoid': Sigmoid activation function -> 1/(1+e^(-z))
                - 'ReLU': Rectified Lienar Unit activation function -> max(0,z)
            The default is 'Sigmoid'.
            
        cost_function : str, optional
            DESCRIPTION: Type of cost function that will be used in Network learning
            VALUES:
                - 'Cross': Cross-Entropy cost function -> C = −1/n ∑_x[yln(a^L)+(1−y)ln⁡(1−a^L)]
                - 'Quad': Quadratic cost function -> C = 1/n ∑_x[(y - a^L)^2]
            The default is 'Cross'.
            
        regularize : float, optional
            DESCRIPTION: In case its different to 0, applies a L2 regularization (or weigth decay) 
                throughout the Network
            VALUES: Non-negative
                - Close to 0: Equivalent of non-regularized case
                - Far from 0: The cost function cares more for keeping weights small than testing accuracy
            The default is 0.

        Returns
        -------
        None.

        '''
        # Defining self attributes
        self.layers = layers
        self.activation_function = activation_function
        self.cost_function = cost_function
        self.regularize = regularize

        # Finding number of neurons 
        self.n = []
        for layer in self.layers:
            if layer[0] == 'Full':
                n_in = layer[1]
            elif layer[0] == 'Conv':
                n_in = int(np.ceil((np.sqrt(n_in) - layer[1] + 1)/layer[3]) *
                           np.ceil((np.sqrt(n_in) - layer[2] + 1)/layer[3]))
            self.n.append(n_in)
        
        # Initializing weigths and biases
        self.weights = []
        for l in range(1, len(self.n)):
            layer = self.layers[l]
            if layer[0] == 'Full':
                self.weights.append(np.random.randn(self.n[l], self.n[l-1])/np.sqrt(self.n[l-1]))
            elif layer[0] == 'Conv':
                self.weights.append([np.random.randn(layer[1],layer[2]) for i in range(layer[4])])
    
#%% Experiments
net = Network(('Full',784),('Conv', 5, 5, 1, 20, 2, 2),('Full',100),('Full',10), activation_function = 'ReLU') 
    
    
    
    
    
    
    
    
