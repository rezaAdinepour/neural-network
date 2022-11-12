import matplotlib.pyplot as plt
import torch
import torch.nn.functional as func
import numpy as np




def sigmoid(x, der=False):
    f = 1 / (1 + np.exp(-x))
    if (der == True):
        f = f * (1 -f)
    return f

def tanh(x, der=False):
    f = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    if (der == True):
        f = 1 - f ** 2
    return f

def linear(x, der=False):
    f = x
    if (der == True):
        f = 1
    return f

def ReLU(x, der=False):
    if (der == True):
        f = np.heaviside(x, 1)
    else :
        f = np.maximum(x, 0)
    return f

def unitStep(x):
    outPut = 1 * (x >= 0)
    #outPut = 1.0 if (x > 0.0) else 0.0
    return outPut



# x = torch.linspace(-5, 5, 200)

# yRelu = torch.relu(x)
# ySigmoid = torch.sigmoid(x)
# yTanh = torch.tanh(x)
# ySoftplus = func.softplus(x)
# yThresh = unitStep(x)


# titles = ['Relu', 'Sigmoid', 'Tanh', 'Soft Plus']
# function = [yRelu, ySigmoid, yTanh, ySoftplus]

# plt.figure('main')



# for i in range(len(function)):
#     plt.subplot(2, 2, i+1)
#     plt.plot(function[i])
#     #plt.grid()
#     plt.title(titles[i])
#     plt.legend(titles[i:], loc='best')
#     plt.xticks([]), plt.yticks([])



# plt.show()


