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
    np.random.seed(123)
    X_train, y_train = load_data('mnist_train.csv')
    X_test, y_test = load_data('mnist_test.csv')
    acc_best = 0
    # 参数查找
    for lr in [1e-2, 2e-2, 1e-1]:
        for wd in [1e-5, 1e-4, 1e-3]:
            for hidden_neuron in [64, 128, 256]:
                print("lr",lr,"wd",wd,"hidden_neuron",hidden_neuron)
                NN = NeuralNetwork(X_train, y_train, X_test, y_test, lr=2e-2, wd=1e-4, hidden_neuron=256) 
                NN.train()
                acc = NN.test(X_test, y_test)
                if acc > acc_best:
                    acc_best = acc
                    print("best group:", "lr",lr,"wd",wd,"hidden_neuron",hidden_neuron, "acc", acc)
                    NN.save()
                    