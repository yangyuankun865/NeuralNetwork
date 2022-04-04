import numpy as np
from network import NeuralNetwork

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
    X_train, y_train = load_data('mnist_train.csv')
    X_test, y_test = load_data('mnist_test.csv')
    
    # 模型测试
    NN = NeuralNetwork(X_train, y_train, X_test, y_test, lr=0.02, wd=1e-4, hidden_neuron=256) 
    NN.load()
    NN.test(X_test, y_test)
    NN.param_plot()
