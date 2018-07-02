import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_breast_cancer
np.random.seed(2)
class NN:
    layer_sizes = None
    weights = []
    biases = []
    layer_count = 0
    z = []
    a = []
    deltas = []
    delta_weights = []
    targets = None
    alpha = 0.1
    def __init__(self, layers):
        self.layer_sizes = layers
        self.layer_count = len(layers)

        self.weights.append(np.arange(1,3).reshape(2, 1))
        self.biases.append(np.arange(1).reshape(1, 1))
        # for i in range(self.layer_count - 1):
        #     self.weights.append(np.random.randn(layers[i], layers[i + 1])*0.01)
        #     self.biases.append(np.zeros((1, layers[i + 1])))
        self.load_data()

    def run(self):
        costs = []
        for i in range(2):
            self.forward()
            self.backward()
        # print(self.a[-1])
        # plt.plot(costs)
        # plt.show()

    def load_data(self):
        self.a.append(np.array(([0, 0], [0, 1], [1, 0], [1, 1])))
        self.targets = np.array([[0], [1], [1], [0]])
        # x, y = load_breast_cancer(return_X_y = True)
        # x = (x - x.mean(axis = 1, keepdims = True))/x.std(axis = 1, keepdims = True)
        # x = x[:10, :]
        # y = y[:10]
        # self.a.append(x)
        # self.targets = y.reshape(10, 1)
    def activation(self, z, type = 'sigmoid'):
        if type == 'relu':
            return z * (z > 0)
        return 1/(1 + np.exp(-z))

    def _activation(self, z, type = 'auto'):
        out = self.activation(z)
        return out*(1 - out)

    def forward(self):
        self.z = []
        self.a = [self.a[0]]
        for i in range(self.layer_count - 1):
            self.z.append(self.a[-1].dot(self.weights[i]) + self.biases[i])
            self.a.append(self.activation(self.z[-1]))
        print(self.a[-1])
    def backward(self):
        # print(self.weights[0].shape)
        # print(self.weights[1].shape)
        # print(((self.a[-1] - self.targets)*self._activation(self.z[-1])).shape)
        # exit()
        self.deltas = [(self.a[-1] - self.targets)*self._activation(self.z[-1])]
        print('\n',self.deltas[-1],'\n\n')
        self.delta_weights = []
        self.delta_biases = []

        for i in reversed(range(0, self.layer_count - 1)):
            self.delta_weights.append(self.a[i].T.dot(self.deltas[-1]))
            self.delta_biases.append(np.sum(self.deltas[-1], axis = 0, keepdims = True))
            self.deltas.append(
                self.deltas[-1].dot(self.weights[i].T) * self._activation(self.z[i-1])
            )

        self.delta_weights = self.delta_weights[::-1]
        self.delta_biases = self.delta_biases[::-1]

        for i in range(self.layer_count-1):
            self.weights[i] = self.weights[i] - self.alpha * self.delta_weights[i]
            self.biases[i] = self.biases[i] - self.alpha * self.delta_biases[i]
            # print(self.weights[i].shape)

    def cost(self, epsilon = 1e-12):
        predictions = np.clip(self.a[-1], epsilon, 1. - epsilon)
        N = predictions.shape[0]
        ce = -np.sum(np.sum(self.targets*np.log(predictions + 1e-9) + (1 - self.targets)*np.log(1 - predictions + 1e-9) ))/N
        return ce




costs = []
nn = NN((2, 4, 1))
nn.run()
# for i in range(10000):
#     cost = nn.forward()
#     if i%100 == 0:
#         costs.append(cost)
#     nn.backward()
#
# print('Done')
# import matplotlib.pyplot as plt
# plt.plot(costs)
# plt.show()
