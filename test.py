import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import ann
from sklearn.model_selection import train_test_split
np.random.seed(1)
def load_data(filename):
    data = np.genfromtxt(filename, delimiter = ',')
    X = data[:, 1:]
    y = data[:, 0].reshape(-1, 1)
    y[y == -1] = 0
    return train_test_split(X, y, test_size=0.2, random_state=1)

X, x_test, y, y_test = load_data('./data_nonlinear.csv')
colors = ['red', 'blue']
plt.scatter(X[:, 0], X[:, 1], c=y[:, :].reshape(-1, ),  cmap=matplotlib.colors.ListedColormap(colors))
plt.show()
model = ann.NN(layers=(2, 3, 10, 3, 1), learning_rate = 0.1, epochs = 1500)
model.load_data(X, y)
model.train(verbose = True)
predictions = model.predict(x_test)
print(model.accuracy(predictions, y_test))
