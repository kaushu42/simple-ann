import matplotlib.pyplot as plt
import numpy as np

np.random.seed(2)

class NN:
    '''
        An aritificial neural network class created using numpy only
    '''
    # Define all the necessary parameters for the network
    layer_sizes = None # A tuple containing the size of all layers i.e the number of neurons in each layer
    layer_count = 0 # To store the number of layers in the nework
    weights = [] # List to store all the weights of the network
    biases = [] # List to store all the biases of the network
    a = [] # List that stores the activations. The first element is set to be the input always.
    z = [] # Activations without applying non-linearity (sigmoid)
    deltas = [] # The gradients with respect to the activations
    delta_weights = [] # The gradients to update the weights
    targets = None # The expected outputs
    alpha = 0 # The learning rate
    epoch = 0 # Number of epochs to perform on the data
    threshold = 0.8 # Minimum confidence to predict as positive
    def __init__(self, layers, learning_rate = 0.01, epochs = 100):
        '''
        Initializes the neural network parameters

        Args:
            layers: A tuple containing the number of neurons in all layers
            learning_rate: The learning rate for gradient ascent
            epochs: The number of iteratons to perform on the data

        Returns:
            None

        Raises:
            None
        '''
        self.layer_sizes = layers
        self.layer_count = len(layers)
        self.alpha = learning_rate
        self.epoch = epochs

        # We need to initialize the weights randomly but we can initialize the biases to zeros
        for i in range(self.layer_count - 1):
            self.weights.append(np.random.randn(layers[i], layers[i + 1]))
            self.biases.append(np.zeros((1, layers[i + 1])))

    def train(self, verbose = False, plot_costs = False):
        '''
        Train the neural network with the loaded data.

        Args:
            verbose: Displays the progress of training if set to True
            plot_costs: Plots the costs with each iteration if set to True

        Returns:
            None

        Raises:
            None

        '''
        costs = []
        for i in range(self.epoch):
            self.forward()
            cost = self.cost()
            costs.append(cost)
            if((i-1) %100 == 0 and verbose):
                print('Iteration: %d\tCost: %f\tAccuracy: %f'%(i- 1, cost, self.train_accuracy()))
            self.backward()
        if plot_costs:
            plt.plot(costs)
            plt.show()

    def load_data(self, X, y):
        '''
        The activation function for the neural network (sigmoid).

        Args:
            X: The input matrix to train the network.
            y: The targets. Encode them as one hot vectors first.
        Returns:
            None

        Raises:
            None
        '''
        self.a.append(X)
        self.targets = y

    @staticmethod
    def activation(z, type = 'sigmoid'):
        '''
        The activation function for the neural network (sigmoid).

        Args:
            z: The input array

        Returns:
            The activation of the input.

        Raises:
            None
        '''
        if type == 'relu':
            return z * (z > 0)
        return 1/(1 + np.exp(-z))

    @staticmethod
    def _activation(z):
        '''
        The derivative of the activation function.

        Args:
            z: The input array.

        Returns:
            The derivative of the activation function for the input.

        Raises:
            None
        '''
        out = NN.activation(z)
        return out*(1 - out)

    def forward(self, X = 'auto'):
        '''
        Perform a forward pass on the neural network to calculate all the activations.

        Args:
            X: The input array on which to perform a forward pass.

        Returns:
            The activations of the output layer.

        Raises:
            None
        '''
        self.z = [] # Clear the previous values
        if str(X) == 'auto':
            self.a = [self.a[0]] # Retain only the input from the previously calculated activations
        else:
            self.a = [X]
        for i in range(self.layer_count - 1):
            self.z.append(self.a[-1].dot(self.weights[i]) + self.biases[i])
            self.a.append(self.activation(self.z[-1]))
        return self.a[-1]

    def backward(self):
        '''
        Perform a backward pass on the network to calculate the gradients with respect to the activations, weights and biases.

        Args:
            None

        Returns:
            None

        Raises:
            None
        '''
        # Calculate the initial delta and clear previous deltas.
        self.deltas = [(self.targets - self.a[-1])*self._activation(self.z[-1])]

        # Clear the previously calculated gradients for weights and biases
        self.delta_weights = []
        self.delta_biases = []

        # Perform backpropagation
        for i in reversed(range(self.layer_count - 1)):
            # dC/dz and dC/db calculation
            self.delta_weights.append(self.a[i].T.dot(self.deltas[-1]))
            self.delta_biases.append(np.sum(self.deltas[-1], axis = 0, keepdims = True))

            # dC/dz for the preceding layer
            self.deltas.append(
                self.deltas[-1].dot(self.weights[i].T) * self._activation(self.z[i-1])
            )

        # The gradients are calculated backwards. So reverse the list to sort the lists in order.
        self.delta_weights = self.delta_weights[::-1]
        self.delta_biases = self.delta_biases[::-1]

        # Perform update on the weights
        for i in range(self.layer_count-1):
            self.weights[i] = self.weights[i] + self.alpha * self.delta_weights[i]
            self.biases[i] = self.biases[i] + self.alpha * self.delta_biases[i]

    def train_accuracy(self):
        predictions = self.a[-1] > self.threshold
        accuracy = np.mean(predictions == self.targets)
        return accuracy

    def predict(self, X):
        predictions = self.forward(X) > self.threshold
        return predictions

    @staticmethod
    def accuracy(predictions, targets):
        return np.mean(predictions == targets)

    def cost(self, epsilon=1e-12):
        """
        Computes the cost between targets predictions. Need to encode the targets as one hot vectors.

        Args:
            epsilon: A small value so that the value of log doesn't go to infinite

        Returns:
            The cross entropy loss for the predictions.
        """
        predictions = np.clip(self.a[-1], epsilon, 1. - epsilon)
        N = predictions.shape[0]
        ce = -np.sum(np.sum(self.targets*np.log(predictions + 1e-9) +
                    (1 - self.targets)*np.log(1 - predictions + 1e-9)))/N
        return ce
