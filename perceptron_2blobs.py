import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
from predict import predict


#information of network
x, y = sklearn.datasets.make_blobs(n_samples=500, n_features=2, centers=[(1, -1), (3, 3)], cluster_std=0.8, center_box=(-10.0, 10.0),
                                   shuffle=True, random_state=None, return_centers=False)

numInput = 2
epochs = 100
learningRate = 0.01
weights = np.random.random(numInput) - 0.5  #generate 2 random number between -0.5 and 0.5
bias = np.random.random(1) - 0.5
#print(bias)
#print(weights)



plt.scatter(x[:, 0], x[:, 1], c=y, cmap='brg')
plt.xlim(-3, 5)
plt.ylim(-3, 5)
plt.waitforbuttonpress()


#train
for ep in range(epochs):
    failCount = 0
    i = 0
    for (data, label) in zip(x, y):
        i += 1
        output = predict(weights, data, bias)
        if(output != label):
            weights += learningRate * (label - output) * data
            bias += learningRate * (label - output)
            failCount += 1

            plt.cla()
            plt.scatter(x[:, 0], x[:, 1], c=y, cmap='brg')
            x_test = np.arange(-3, 5.1, 0.1)
            y_test = (-1 / weights[1]) * (bias + weights[0] * x_test) # bias + W1*x_test + W2*y_test = 0 -> y_test = (-1 / W2) * (W0 + W1*x_test)
            plt.plot(x_test, y_test, color='black')
            plt.xlim(-3, 5)
            plt.ylim(-3, 5)
            plt.text(-3, 4.5, 'epoch|iter = {:2d}|{:2d}'.format(ep, i), fontdict={'size': 16, 'color': 'red'})
            plt.pause(0.01)


    if(failCount == 0):
        plt.waitforbuttonpress()
        break


plt.show()