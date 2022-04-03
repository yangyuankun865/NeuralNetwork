import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, X, y, X_test, y_test, batch = 1, lr = 1e-3, lr_decay = 0.1, wd = 1e-4, hidden_neuron=256, epochs = 6):
        self.input = X 
        self.target = y
        self.input_test = X_test
        self.target_test = y_test
        self.batch = batch
        self.epochs = epochs
        self.hidden_neuron = hidden_neuron
        self.lr = lr
        self.lr_decay = lr_decay
        self.wd = wd
        
        self.x = self.input[:self.batch] # batch input 
        self.y = self.target[:self.batch] # batch target value
        self.test_interval = 10000
        self.loss_train = []
        self.loss_test = []
        self.acc_train = []
        self.acc_test = []
        
        self.init_weights()
      
    def init_weights(self): # 参数初始化
        self.W1 = np.random.randn(self.input.shape[1],self.hidden_neuron)
        self.W2 = np.random.randn(self.W1.shape[1],self.y.shape[1])

        self.b1 = np.random.randn(self.W1.shape[1],)
        self.b2 = np.random.randn(self.W2.shape[1],)

    def ReLU(self, x): # 激活函数
        return np.maximum(0,x)

    def dReLU(self,x):
        return 1 * (x > 0) 
    
    def softmax(self, z):
        z = z - np.max(z, axis = 1).reshape(z.shape[0],1)
        return np.exp(z) / np.sum(np.exp(z), axis = 1).reshape(z.shape[0],1)
    
    def shuffle(self):
        idx = [i for i in range(self.input.shape[0])]
        np.random.shuffle(idx)
        self.input = self.input[idx]
        self.target = self.target[idx]
        
    def feedforward(self): # loss 计算
        assert self.x.shape[1] == self.W1.shape[0]
        self.z1 = self.x.dot(self.W1) + self.b1
        self.a1 = self.ReLU(self.z1)

        assert self.a1.shape[1] == self.W2.shape[0]
        self.z2 = self.a1.dot(self.W2) + self.b2
        self.a2 = self.softmax(self.z2)
        self.error = self.a2 - self.y
        
    def backprop(self):  
        dcost = (1/self.batch)*self.error
        # 梯度计算
        DW2 = np.dot(dcost.T,self.a1).T
        DW1 = np.dot((np.dot((dcost),self.W2.T) * self.dReLU(self.z1)).T,self.x).T

        db2 = np.sum(dcost,axis = 0)
        db1 = np.sum(np.dot((dcost),self.W2.T) * self.dReLU(self.z1),axis = 0)
        
        assert DW2.shape == self.W2.shape
        assert DW1.shape == self.W1.shape
        
        assert db2.shape == self.b2.shape
        assert db1.shape == self.b1.shape 
        # 反向传播更新参数  使用L2正则化
        self.W2 = self.W2 - self.lr * (DW2 + self.W2 * self.wd)
        self.W1 = self.W1 - self.lr * (DW1 + self.W1 * self.wd)
        
        self.b2 = self.b2 - self.lr * (db2 + self.b2 * self.wd)
        self.b1 = self.b1 - self.lr * (db1 + self.b1 * self.wd)

    def train(self):
        iteration = 0

        l, l_test, acc, acc_test  = 0, 0, 0, 0
        for epoch in range(self.epochs):
            if (epoch) % 2 == 0 and epoch != 0:
                self.lr = self.lr * self.lr_decay
            self.shuffle()
            # 使用随机梯度下降SGD时,即self.batch=1, 选取单个样本进行反向传播更新参数
            # 此处同样实现MBGD，可选取self.batch>1多个样本反向传播
            for batch in range(self.input.shape[0]//self.batch-1):
                start = batch*self.batch
                end = (batch+1)*self.batch
                self.x = self.input[start:end] 
                self.y = self.target[start:end]
                # print("self.x",self.x.shape, "self.y",self.y.shape)
                self.feedforward()
                l += -(self.y*np.log(self.a2+1e-8)).sum() + self.wd * ((self.W1*self.W1).sum() + (self.W2*self.W2).sum() + np.inner(self.b1,self.b1) + np.inner(self.b2,self.b2))
                self.backprop()
                acc += np.count_nonzero(np.argmax(self.a2,axis=1) == np.argmax(self.y,axis=1)) / self.batch
                iteration +=1
                if iteration % self.test_interval == 0:
                    for batch in range(self.input_test.shape[0]//self.batch-1):
                        start = batch*self.batch
                        end = (batch+1)*self.batch
                        self.x = self.input_test[start:end]
                        self.y = self.target_test[start:end]
                        self.feedforward()
                        l_test += -(self.y*np.log(self.a2+1e-8)).sum() + self.wd * ((self.W1*self.W1).sum() + (self.W2*self.W2).sum() + np.inner(self.b1,self.b1) + np.inner(self.b2,self.b2))
                        acc_test += np.count_nonzero(np.argmax(self.a2,axis=1) == np.argmax(self.y,axis=1)) / self.batch


                    self.loss_train.append(l/self.test_interval)
                    self.acc_train.append(acc*100/self.test_interval)
                    self.loss_test.append(l_test/(self.input_test.shape[0]//self.batch))
                    self.acc_test.append(acc_test*100/(self.input_test.shape[0]//self.batch))
                    print("lr", self.lr, "loss", l/self.test_interval, "acc",acc*100/self.test_interval,
                    "loss_test",l_test/(self.input_test.shape[0]//self.batch), "acc_test", acc_test*100/(self.input_test.shape[0]//self.batch))
                    l, l_test, acc, acc_test  = 0, 0, 0, 0
            
    def save(self):            
        np.savez('weight.npz', W1=self.W1,  W2=self.W2, b1=self.b1,  b2=self.b2)
    
    def load(self):    
        data = np.load('weight.npz')    
        self.W1,  self.W2, self.b1,  self.b2 = data['W1'], data['W2'], data['b1'], data['b2'], 
    
    def plot(self, name):
        plt.figure(dpi = 125)
        iteration = np.arange(len(self.loss_train))*self.test_interval+self.test_interval
        plt.plot(iteration, self.loss_train, label='train')
        plt.plot(iteration, self.loss_test, label='test')
        plt.legend()
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.savefig('./plot/'+str(name)+'loss.png')
    
    def acc_plot(self, name):
        plt.figure(dpi = 125)
        iteration = np.arange(len(self.loss_train))*self.test_interval+self.test_interval
        plt.plot(iteration, self.acc_train, label='train')
        plt.plot(iteration, self.acc_test, label='test')
        plt.legend()
        plt.xlabel("Iterations")
        plt.ylabel("Accuracy")
        plt.savefig('./plot/'+str(name)+'accuracy.png')

    def param_plot(self):
        W1_plot = self.W1
        print("self.W1", self.W1.shape)
        W1_plot = W1_plot.reshape(28, 28, 1, -1).transpose(3, 0, 1, 2)
        print("W1_plot",W1_plot.shape)
        plt.figure(figsize = (16, 16))
        for index in range(W1_plot.shape[0]):
            plt.subplot(16, 16, index + 1)
            plt.imshow(W1_plot[index,:,:], cmap=plt.cm.gray)
            plt.axis('off')
            plt.savefig('./plot/parameterW1.png')


        W2_plot = self.W2
        print("self.W2", self.W2.shape)
        W2_plot = W2_plot.reshape(16, 16, 1, -1).transpose(3, 0, 1, 2)
        print("W2_plot",W2_plot.shape)
        plt.figure(figsize = (2, 5))
        for index in range(W2_plot.shape[0]):
            plt.subplot(2, 5, index + 1)
            plt.imshow(W2_plot[index,:,:], cmap=plt.cm.gray)
            plt.axis('off')
            plt.savefig('./plot/parameterW2.png')


        
    def test(self, xtest, ytest):
        self.x = xtest
        self.y = ytest
        self.feedforward()
        acc = np.count_nonzero(np.argmax(self.a2,axis=1) == np.argmax(self.y,axis=1)) / self.x.shape[0]
        print("Accuracy:", 100 * acc, "%")
        return acc
    
    