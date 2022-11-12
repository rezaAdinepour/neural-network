import numpy as np
from activation_functions import*


def mlpTrain(xTrain, yTrain, h1=4, h2=4, eta=0.1, epochs=100):

    w1 = 2 * np.random.rand(h1, xTrain.shape[1]) - 1.0
    b1 = np.random.rand(h1)

    w2 = 2 * np.random.rand(h2, h1) - 1.0
    b2 = np.random.rand(h2) #- 1

    wOut = np.random.rand(h2) - 1.0
    bOut = np.random.rand(1) #- 1

    loss = 0
    train_loss = []

    for ep in range(epochs):
        print('epoch ', ep)
        for i in range(0, xTrain.shape[0]-1):
            x = xTrain[i]
            
            o1 = ReLU(np.dot(w1, x) + b1)
            o2 = ReLU(np.dot(w2, o1) + b2)
            y = sigmoid(np.dot(wOut, o2) + bOut)

            deltaOut = 2 * (y - yTrain[i]) * sigmoid(y, der=True)
            delta2 = deltaOut * wOut * ReLU(o2, der=True)
            delta1 = np.dot(delta2, w2) * ReLU(o1, der=True)

            wOut -= eta * deltaOut * o2
            bOut -= eta * deltaOut

            w2 -= eta * np.kron(delta2, o1).reshape(h2, h1)
            b2 = b2 - eta * delta2

            w1 -= eta * np.kron(delta1, x).reshape(h1, x.shape[0])
            b1 -= eta * delta1

            loss += (y - yTrain[i]) ** 2
            train_loss.append(y)

        loss /= xTrain.shape[0]

        if(ep % 10 == 0):
            o1 = ReLU(np.dot(w1, xTrain.T) + b1.reshape(b1.shape[0], 1))
            o2 = ReLU(np.dot(w2, o1) + b2.reshape(b2.shape[0], 1))
            prediction = sigmoid(np.dot(wOut, o2) + bOut)

            plt.cla()
            plt.axis([-1, 9, -1.5, 1.5])
            plt.scatter(xTrain, yTrain, color='blue')
            plt.scatter(xTrain, prediction, color='green', alpha=0.8)
            plt.text(6.0, -1.0, 'epoch = %d' %ep, fontdict={'size': 14, 'color': 'red'})
            plt.text(6.0, -1.25, 'loss = %.4f' %loss, fontdict={'size': 14, 'color': 'red'})
            plt.pause(0.01)
        if(loss < 0.001):
            break
    return w1, b1, w2, b2, wOut, bOut, loss


x = torch.linspace(0, 8, 9)
#print(X)
x = torch.unsqueeze(x, dim=1) #transpose x
#print(x)
xTest = torch.unsqueeze(torch.linspace(0, 8, 800), dim=1)
y = torch.tensor([0, 0.84, 0.91, 0.14, -0.77, -0.96, -0.28, 0.66, 0.99])
EPOCH = 2000


print('*'*10, '2 Layer MLP From Scratch', '*'*10)
w1, b1, w2, b2, wOut, bOut, mu = mlpTrain(x.data.numpy(), y.data.numpy(), h1=4, h2=4, eta=0.1, epochs=EPOCH)