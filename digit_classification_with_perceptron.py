import numpy as np
import matplotlib.pyplot as plt


def predict(inputs, weights):
    sum = weights[0] + np.dot(inputs, weights[1:])
    activation = (sum > 0.5) * 1.0
    return activation



inputs = np.array( [ 
                    [1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 0, 1, 1, 1, 1, 1],
                    [1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 0, 0, 1, 1, 1, 0, 1, 1],
                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 0, 1, 1, 1, 0, 0, 1, 1, 1],
                    [1, 0, 1, 1, 1, 1, 1, 0, 1, 1],
                    [0, 1, 1, 1, 1, 1, 1, 0, 1, 1],
                    [1, 0, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 0, 1, 0, 0, 0, 1, 0, 1, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [1, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 0, 1, 1, 0, 1, 1],
                    [1, 1, 1, 1, 0, 1, 1, 0, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1] ] )


for i in range(10):
    sample = 1 - inputs[:, i].reshape(5, 3)
    plt.subplot(3, 4, i+1)
    plt.imshow(sample)

#plt.show()

targets = np.array( [
                    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1] ] )


inputNeurons = inputs.shape[0]
outputNeurons = targets.shape[0]
#print(inputNeurons)
#print(outputNeurons)

epochs = 100
learningRate = 0.01

weight = np.random.rand(inputNeurons + 1, outputNeurons) - 0.5 #create 16*10 matrix
#print(weight)


for ep in range(epochs):
    failCount = 0
    for (input, label) in zip(inputs.T, targets):
        prediction = predict(input, weight)
        if(np.sum(np.abs(label - prediction)) != 0):
            weight[1:] += learningRate * (label - prediction) * input.reshape(input.shape[0], 1)
            weight[0] += learningRate * (label - prediction)
            failCount += 1
    print('epoch {}: fail count {}'.format(ep, failCount))
    if(failCount == 0):
        break

for (input, label) in zip(inputs.T, targets):
    print('input = ', input, end=' ')
    prediction = predict(input, weight)
    print('output = ', prediction)
    print('target = ', label)