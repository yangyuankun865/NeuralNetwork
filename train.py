import numpy as np
from network import NeuralNetwork
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--lr', default=2e-2)
parser.add_argument('--wd', default=1e-4)
parser.add_argument('--hidden_neuron', default=256)
args = parser.parse_args()

args = parser.parse_args()
# pauser
def load_data(path):
    def one_hot(y):
        table = np.zeros((y.shape[0], 10))
        for i in range(y.shape[0]):
            table[i][int(y[i][0])] = 1 
        return table

    def normalize(x): 
        x = x / 255
        return x 

    data = np.loadtxt('{}'.format(path), delimiter = ',')
    return normalize(data[:,1:]),one_hot(data[:,:1])

if __name__ == "__main__":
    np.random.seed(123)
    X_train, y_train = load_data('mnist_train.csv')
    X_test, y_test = load_data('mnist_test.csv')
    
    # 模型训练
    for wd in [1e-5, 1e-4, 1e-3]:
        NN = NeuralNetwork(X_train, y_train, X_test, y_test, lr=args.lr, wd=args.wd, hidden_neuron=args.hidden_neuron) 
        NN.train()
        NN.plot(wd)
        NN.acc_plot(wd)

