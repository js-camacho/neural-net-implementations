"""
Created on Wed Apr  1 16:29:32 2020

@author: johan

This is a new implementation of the general Neural Netowrk code in the FirstNeuralNetwork.py file
using the same general class Network and same MNIST data sets, but this time there are some new
features to improve the overall Neural Network performance. These include:
    
    - Fully matrix based approach for backpropagation:
        + Improves the computational time in general
        * These approach create matrices for each mini-batch to train the Network over all training
          examples simultaniously rather than iterating over each training example vector.
          
    - Cross-Entropy cost function:
        + Reduce learning slowdown due to saturation in the output neurons (when output neurons take values
          near 0 and 1 their learning slows down because the \sigma_prime term becomes almost 0, reducing 
          the gradient)
        * The cross-entropy function is a new cost function, namely:
            C=−1/n ∑_x[yln(a^L)+(1−y)ln⁡(1−a^L)]
          These alters the equations (BP1) and (BP4) for the output layer

    - L2 regularization (or weight decay):
        + Reduce overfitting considerably (overfitting is when a nueral net stops recognizing abstract patterns
          in training data and starts memorizing the data set, this means that the accuracy % on the test data
          will be stuck while the accuracy % in the training data continues to grow, increasing the difference
          between these two)
        * This regularization includes an additional term in the cost function, namely:
            C = C_0 + \lambda/(2n)*∑_x(w^2)     where lambda is called the regularization parameter
          This implies that the cost function will create a trade off between improving accuracy and keeping the
          weights small.
          These alters the equation learning rule of the weights, namely:
              w <- (1 - \eta\lambda/n)w - \eta/m ∑_x(∂C_x/∂w)
              
    - Improved weight initialization:
        + Helps preventing early neurons saturation in hidden layers, therefore, learning will speed up in
          early epochs of training
        * Instead of initializing weights using Normal(0, 1), they will be initialized using 
          Normal(0, 1/sqrt(number_input_neurons))
          
    - Early stop for Stochastic Gradient Descent:
        + Reduce overfitting/overtraining and avoid defining the epochs parameter
        * Creates a rule to stop the SGD when a maximum number of epochs with no improvement is surpassed. These
        number of epochs is defined as a parameter <<early_stop>> in Network.SGD
        
    - Learning rate schedule:
        + Improves the learning in different stages of SGD, making the gradient step size gradually smaller
        * Given an initial value of learning rate, the SGD method divides the learning rate by some factor 
        after a given number of epoch wiht no improvement in accuracy. The parameter to use the learning 
        rate schedule is a tuple that contains: (epochs_with_no_improvement, dividing_factor, maximum_divisions)
        
    - Momentum-based gradient descent:
        + Improves learning speed based on the last gradient
        * There are new variables called velocities, which store the gradient step at each point of learning, then
        this value is reduced by a parameter 'mu' and then the weights are modified accoring to this velocities.
        This creates a sort of memory that increase the rate of learning when the gradient is similar as before,
        and this acceleration is controlled by the parameter 'mu'

"""

#%% Importing Libraries

# Standard library
import random
from datetime import datetime

# Third-party libraries
import numpy as np
import pandas as pd
import PIL as pil
import matplotlib.pyplot as plt

#%% Neural Network Implementation

class Network(object):

    def __init__(self, sizes, weights_init = 'Improved'):
        """The list <<sizes>> contains the number of neurons in the respective layers of the network.  
        For example, if the list was [2, 3, 1] then it would be a three-layer network, with the first layer 
        containing 2 neurons, the second layer 3 neurons, and the third layer 1 neuron.  
        
        The <<weights_init>> define the method for initializing weights: 'Normal' = N(0, 1) and 
        'Improved' = N(0, 1/sqrt(n_input_neurons))
        
        The biases for the network are initialized randomly, using a Gaussian (Normal) distribution 
        with mean 0, and variance 1.  
        
        The velocities are the variables used in momentum-based gradient descent, are initialized in 0 and
        have the same shape as weights
        
        Note that the first layer is assumed to be an input layer, and by  convention we won't set any biases 
        for those neurons, since biases are only ever used in computing the outputs from later layers."""
        
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        if weights_init == 'Improved':
            self.weights = [np.random.randn(y, x)/np.sqrt(x) for x, y in zip(sizes[:-1], sizes[1:])]    
        else:
            self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        self.velocities = [np.zeros((y, x)) for x, y in zip(sizes[:-1], sizes[1:])]   # Used for momentum-based GD

    def feedforward(self, a, activ_funct = 'Sigmoid'):
        """
        Return the output of the next layer of the network if <<a>> are the activations of a previous layer.
        It is important that the input is a (n, 1) numpy array, not a (n, ) vector:
            input:  [0, 1]              Error
                    np.array([0,1])     Error
                    np.array([[0],[1]]) OK
        """
        if activ_funct == 'Sigmoid':
            for b, w in zip(self.biases, self.weights):
                a = sigmoid(np.dot(w, a)+b)
            return a
        elif activ_funct == 'ReLU':
            for b, w in zip(self.biases, self.weights):
                a = ReLU(np.dot(w, a)+b)
            return a

    def SGD(self, training_data, epochs, mini_batch_size, eta, eta_schedule = (5, 2, 4), mu = 0, test_data = None,
            method = 'M', cost_fun = 'CE', reg_parameter = 0, early_stop = 999, activ_funct = 'Sigmoid'):
        """Train the neural network using mini-batch Stochastic Gradient Descent.  
        - The <<training_data>> is a list of tuples (x, y) representing the training inputs (x) and the desired 
        outputs (y). 
        - The <<method>> must be either V for vectorial backpropagation or 'M' for matrix 
        backpropagation. 
        - The <<cost_fun>> defines the cost function, can be CE for cross-entropy or Q for quadratic
        - The <<reg_parameter>> is the parameter that is applied to the learning equation of the weights according
        to the L2 regularization.
        - <<early_stop>> defines the number of permited epochs accepted without improvement in testing accuracy
        before shutting down the SGD, if None is provided, then the number of epochs is fixed by the 
        parameter <<epochs>>
        - The parameter <<eta_schedule>> implements a learning rate schedule. Given an initial value of learning 
        rate, the SGD method divides the learning rate by some factor after a given number of epoch wiht no 
        improvement in accuracy. The parameter to use the learning rate schedule is a tuple that contains: 
        (epochs_with_no_improvement, dividing_factor, maximum_divisions)
        - The parameter <<mu>> implements a Momentum-based gradient descent. <<mu>> controlls the "friction"
         which tends to gradually reduce the "velocity" coefficient. <<mu>> ranges from 0 to 1. When <<mu>> = 0
         there is no friction which means the velocity controlls absolutely the weights (which can lead to
         overshooting or going in wrong direction). When <<mu>> = 0 there is a lot of friction (same as SGD).
         So, for momentum-based gradient descent to work <<mu>> must be tuned like <<eta>> between 0 and 1.
        
        The other non-optional parameters are self-explanatory.  
        
        If <<test_data>> is provided then the network will be evaluated against the test data after each epoch, 
        and partial progress printed out.  This is useful for tracking progress, but slows things down 
        substantially."""
        
        if test_data: 
            n_test = len(test_data)
        
        n = len(training_data)
        start = datetime.now()      # Starting timer 
        no_improve = 0               # Count the number of epchs without improving the best testing accuracy
        best_te_acc = 0
        eta_epochs, eta_divide, eta_max_divisions = eta_schedule
        
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            
            # Backpropagation
            #reg_parameter = reg_parameter/n     # Diving the regularization parameter by the number of training examples
            if method == 'V':
                for mini_batch in mini_batches:
                    self.update_mini_batch(mini_batch, eta, mu, cost_fun, activ_funct)
            elif method == 'M':
                for mini_batch in mini_batches:
                    self.update_mini_batch_full_matrix(mini_batch, eta, mu, mini_batch_size, 
                                                       cost_fun, reg_parameter, activ_funct)  
                    
            stop = datetime.now() - start  # Stopping timer 
            if test_data:   
                stop = datetime.now() - start  # Stopping timer
                print ('\nEpoch ',j,':-----------------------')
                #tr_acc =  self.evaluate(training_data2, activ_funct)
                #print('Training data accuracy:',tr_acc,'/', n,'\t('+str(round(tr_acc/n*100,2))+'%)')
                te_acc = self.evaluate(test_data, activ_funct)
                print('Test data accuracy:',te_acc,'/', n_test,'\t('+str(round(te_acc/n_test*100, 2))+'%)')
                print('Time elapsed:',stop)
                print('---------------------------------')
                
                if te_acc > best_te_acc:
                    no_improve = 0
                    best_te_acc = te_acc
                else:
                    no_improve += 1
                    # Checking condition for early stoping SGD
                    if no_improve >= early_stop:
                        print('\nEarly Stop **************************')
                        print('Best recorded accuracy:('+str(round(best_te_acc/n_test*100, 2))+'%)')
                        print('*************************************')
                        break  
                    # Managing learning rate schedule
                    if no_improve >= eta_epochs:
                        eta /= eta_divide
                        eta_max_divisions -= 1
                        
                        if eta_max_divisions == 0:
                            eta_divide = 1          # This value prevents eta from changing from now on 
                        else:
                            print('\nEta changed to:',eta)
                
            else:
                print ('Epoch',j,'complete\tTime elapsed:',stop)
        
               
        stop = datetime.now() - start  # Stopping timer        
        print('Total time elpased:', stop)

    def update_mini_batch(self, mini_batch, eta, mu, cost_fun = 'CE', reg_parameter = 0, activ_funct = 'Sigmoid'):
        """Update the network's weights and biases by applying gradient descent using backpropagation to a single 
        mini batch.
        
        The <<mini_batch>> is a list of tuples (x, y), and <<eta>> is the learning rate."""
        
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y, cost_fun, activ_funct)
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        
        # Momentum-based gradient descent: update velocities (self.velocities, mu)
        self.velocities = [(mu*v) - (eta/len(mini_batch))*nw for v, nw in zip(self.velocities, nabla_w)]
        # L2 regularization (reg_parameter)
        self.weights = [(1 - reg_parameter*eta)*w + v for w, v in zip(self.weights, self.velocities)]  
        self.biases = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]
        
    def update_mini_batch_full_matrix(self, mini_batch, eta, mu, mini_batch_size, cost_fun = 'CE',
                                      reg_parameter = 0, activ_funct = 'Sigmoid'):
        """Update the network's weights and biases by applying gradient descent using backpropagation to a single 
        mini batch.
        
        The FULL MATRIX approach indicates that instead of taking one vector for each training example,
        all training examples are stored as a matrix and the operations are modified to deal with these matrices.
        
        The <<mini_batch>> is a list of tuples (x, y), and <<eta>> is the learning rate."""
        
        # Converting the mini-batch to a pair of matrices X and Y
        X = mini_batch[0][0]
        Y = mini_batch[0][1]
        for x, y in mini_batch[1:]:
            X = np.concatenate((X,x), axis = 1)
            Y = np.concatenate((Y,y), axis = 1)
        
        # Using the Backpropagation Full Matrix method to calculate the gradient
        nabla_b, nabla_w = self.backprop_full_matrix(X, Y, mini_batch_size, cost_fun, activ_funct)
        
        # Updating the weights and biases of the Neural Network
        
        # Momentum-based gradient descent: update velocities (self.velocities, mu)
        self.velocities = [(mu*v) - (eta/len(mini_batch))*nw for v, nw in zip(self.velocities, nabla_w)]
        # L2 regularization (reg_parameter)
        self.weights = [(1 - reg_parameter*eta)*w + v for w, v in zip(self.weights, self.velocities)]    
        self.biases = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y, cost_fun = 'CE', activ_funct = 'Sigmoid'):
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
            if activ_funct == 'Sigmoid':
                activation = sigmoid(z)
            elif activ_funct == 'ReLU':
                activation = ReLU(z)
            activations.append(activation)
        # Backward pass
        if cost_fun == 'CE':
            delta = self.cost_derivative(activations[-1], y)                                # (BP1)
        elif cost_fun == 'Q':
            if activ_funct == 'Sigmoid':
                delta = self.cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])    # (BP1)
            elif activ_funct == 'ReLU':
                delta = self.cost_derivative(activations[-1], y) * ReLU_prime(zs[-1])       # (BP1)
        nabla_b[-1] = delta                                                                 # (BP3)
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())                            # (BP4)
        # Note that the variable l in the loop below is used a little
        # differently to the notation in Chapter 2 of the book.  Here,
        # l = 1 means the last layer of neurons, l = 2 is the
        # second-last layer, and so on.  It's a renumbering of the
        # scheme in the book, used here to take advantage of the fact
        # that Python can use negative indices in lists.
        for l in range(2, self.num_layers):
            z = zs[-l]
            if activ_funct == 'Sigmoid':
                delta = np.dot(self.weights[-l+1].transpose(), delta) * sigmoid_prime(z)    # (BP2)
            elif activ_funct == 'ReLU':
                delta = np.dot(self.weights[-l+1].transpose(), delta) * ReLU_prime(z)       # (BP2)
            nabla_b[-l] = delta                                                             # (BP3)
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())                      # (BP4)
        return (nabla_b, nabla_w)
    
    def backprop_full_matrix(self, X, Y, mini_batch_size, cost_fun = 'CE', activ_funct = 'Sigmoid'):
        """Return a tuple <<(nabla_b, nabla_w)>> representing the gradient for the cost function C_x. 
        
        The FULL MATRIX approach indicates that instead of taking one vector for each training example,
        all training examples are stored as a matrix and the operations are modified to deal with these matrices.
        
        <<cost_fun>> can be CE for cross-entropy or Q for quadratic
        
        <<nabla_b>> and <<nabla_w>> are layer-by-layer lists of numpy arrays, similar to <<self.biases>> and 
        <<self.weights>>."""
        
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        
        # Feedforward -------------------------------------------------------------------------------------------
        Zs = []
        activation = X                              # Activations are now matrices
        activations = [X]
        for b, w in zip(self.biases, self.weights):
            Z = np.dot(w, activation)+b
            Zs.append(Z)
            if activ_funct == 'Sigmoid':
                activation = sigmoid(Z)             # Sigmoid function is applied element-wise to the matrix Z
            elif activ_funct == 'ReLU':
                activation = ReLU(Z)                # ReLU function is applied element-wise to the matrix Z
            activations.append(activation)
    
        # Calculation the error matrix of the last layer
        if cost_fun == 'CE':
            delta = self.cost_derivative(activations[-1], Y)                                # (BP1)
        elif cost_fun == 'Q':
            if activ_funct == 'Sigmoid':
                delta = self.cost_derivative(activations[-1], Y) * sigmoid_prime(Zs[-1])    # (BP1)
            elif activ_funct == 'ReLU':
                delta = self.cost_derivative(activations[-1], Y) * ReLU_prime(Zs[-1])       # (BP1)
        e = np.ones((mini_batch_size,1))
        nabla_b[-1] = np.dot(delta,e)
        #nabla_b[-1] = delta.sum(axis = 1).reshape(self.biases[-1].shape)                   # (BP3)
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())                            # (BP4)
        
        # Backward Pass -----------------------------------------------------------------------------------------    
        for l in range(2, self.num_layers):
            Z = Zs[-l]
            if activ_funct == 'Sigmoid':
                delta = np.dot(self.weights[-l+1].transpose(), delta) * sigmoid_prime(Z)    # (BP2)
            elif activ_funct == 'ReLU':
                delta = np.dot(self.weights[-l+1].transpose(), delta) * ReLU_prime(Z)       # (BP2)    
            nabla_b[-l] = np.dot(delta,e)
            #nabla_b[-l] = delta.sum(axis = 1).reshape(self.biases[-l].shape)               # (BP3)
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())                      # (BP4)
        return (nabla_b, nabla_w)
        
        
    def evaluate(self, test_data, activ_funct):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = [(np.argmax(self.feedforward(x, activ_funct)), y) for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, a, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (a-y)

# Squishification function and its derivate
def sigmoid(z):
    """The sigmoid function."""
    return 1.0/(1.0+np.exp(-z))

def ReLU(z):
    '''
    Rectified Linear Unit Function: max(0,z)
    '''
    return np.maximum(z, 0, z)

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    return sigmoid(z)*(1-sigmoid(z))

def ReLU_prime(z):
    '''
    Derivative of the Rectified Linear Unit
    0 if z is less than 0 and 1 otherwise
    '''
    z[z<0] = 0
    z[z>0] = 1
    return z
    

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

#%% Experiments

net = Network([784, 100, 10], weights_init = 'Improved')
net.SGD(training_data, epochs = 100, mini_batch_size = 20, eta = 0.1, mu = 0.85, eta_schedule = (3,2,6),
        test_data = testing_data, method = 'M', cost_fun = 'CE', reg_parameter = 0.0001, early_stop = 5,
        activ_funct = 'Sigmoid')

#%% Example printer
def print_example(row = random.randint(0,10000), failed = False):
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
    
print_example(failed = True)
    
#%%
#*******************************************
# Record: 98.30 %
#*******************************************

#%% Read image

# Funtion for reading an image to a numpy matrix

image = pil.Image.open(r"Sample.PNG").convert('L')
image = image.resize((28,28), pil.Image.ANTIALIAS)
image.save('Sample 28x28.png')
ar = np.array(image)
print(ar.shape)

# Sample image

sample_image = [(1,0)]
pixels = []
for i in range(28):
    for j in range(28):
        pixels.append(ar[i][j])
pixels = np.array(pixels).reshape((784,1))
pixels = (255 - pixels)/255

plt.matshow(pixels.reshape((28,28)))
plt.show()

print('********************')    
print('Network Guess:',np.argmax(net.feedforward(pixels)),'\nReal Value:', 3)
print('********************')

#%% End of file