#from array import array
#import numpy as np


'''inputs = [1, 2, 3]
weights = [0.2, 0.8, -0.5]
bias = 2
output = ( inputs[0] * weights[0] +
           inputs[1] * weights[1] +
           inputs[2] * weights[2] + bias )

print(output)'''








'''inputs = [1, 2, 3, 2.5]
weights = [ [0.2, 0.8, -0.5, 1],
            [0.5, -0.91, 0.26, -0.5],
            [-0.26, -0.27, 0.17, 0.87] ]
biases = [2, 3, 0.5]

layerOutputs = []
for (neuronWeights, neuronBias) in zip(weights, biases):
    neuronOutput = 0
    for (nInput, weight) in zip(inputs, neuronWeights):
        neuronOutput += nInput * weight
    neuronOutput += neuronBias
    layerOutputs.append(neuronOutput)

print(layerOutputs)'''







'''inputs = [1.0, 2.0, 3.0, 2.5]
weights = [0.2, 0.8, -0.5, 1.0]
bias = 2.0
output = np.dot(inputs, weights) + bias
print(output)'''







'''inputs = np.array([1.0, 2.0, 3.0, 2.5])
weights = np.array([ [0.2, 0.8, -0.5, 1],
                     [0.5, -0.91, 0.26, -0.5],
                     [-0.26, -0.27, 0.17, 0.87] ])
biases = np.array([2.0, 3.0, 0.5])

layerOutputs = np.dot(weights, inputs) + biases
print(layerOutputs)'''
