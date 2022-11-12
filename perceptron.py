import numpy as np
import matplotlib.pyplot as plt
from predict import predict
from plot_boundary import plot_decision_boundary_wm


#information of network
datas = np.array([ [1, 1], [1, -1], [0, -1], [-1, -1], [-1, 1], [0, 1]]) #input data
#print(data)
labels = np.array([1, 1, 1, 0, 0, 0])
numInput = 2
epochs = 100
learningRate = 0.1
weights = np.random.random(numInput) - 0.5  #generate 2 random number between -0.5 and 0.5
bias = np.random.random(1) - 0.5
#print(bias)
#print(weights)


firstClass = datas[0:3, :]
secondClass = datas[3:, :]
print(firstClass)
#print(secondClass)


plt.scatter(firstClass[:, 0], firstClass[:, 1], s = 30, color = 'r', alpha = 0.9)  
plt.scatter(secondClass[:, 0], secondClass[:, 1], s = 30, color = 'b', alpha = 0.9)
plt.xlim(-2, 2)
plt.ylim(-2, 2)
print(firstClass[:, 0])


#train
for ep in range(epochs):
    failCount = 0
    for (data, label) in zip(datas, labels):
        output = predict(weights, data, bias)
        if(output != label):
            weights += learningRate * (label - output) * data
            bias += learningRate * (label - output)
            failCount += 1
        
        plt.cla()
        x = np.arange(-2, 2, 0.1)
        plt.scatter(firstClass[:, 0], firstClass[:, 1], s = 30, color = 'r', alpha = 0.9)  
        plt.scatter(secondClass[:, 0], secondClass[:, 1], s = 30, color = 'b', alpha = 0.9)
        y = (-bias - weights[0] * x) / weights[1]
        y = (-1 / weights[1]) * (bias + weights[0] * x) # bias + W1*x + W2*y = 0 -> y = (-1 / W2) * (W0 + W1*x)
        plt.plot(x, y, color='black')
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        plt.text(0.8, -1.7, 'epoch = {:2d}'.format(ep), fontdict={'size': 16, 'color': 'red'})
        plt.pause(0.01)
        
    if(failCount == 0):
        plt.waitforbuttonpress()
        plot_decision_boundary_wm()
        break

for (data, label) in zip(datas, labels):
    print('input = ', data, end=' ')
    output = predict(weights, data, bias)
    print('output = ', output)


plt.show()