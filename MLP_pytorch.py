import torch
import matplotlib.pyplot as plt


x = torch.linspace(0, 8, 9)
#print(X)
x = torch.unsqueeze(x, dim=1) #transpose x
#print(x)
xTest = torch.unsqueeze(torch.linspace(0, 8, 800), dim=1)
y = torch.tensor([0, 0.84, 0.91, 0.14, -0.77, -0.96, -0.28, 0.66, 0.99])
#print(xTest)

plt.plot(x, y, 'r*')

#create network with 1 input ans 5 neurons in hidden layer and 1 output
network = torch.nn.Sequential (
                                torch.nn.Linear(1, 5),
                                torch.nn.Sigmoid(),
                                torch.nn.Linear(5, 1) )

#print(network)

y1 = network(xTest)
plt.plot(x, y, 'r*', xTest, y1.detach().numpy(), 'b--')
plt.axis([-1, 9, -1.5, 1.5])
plt.title('response with initial weights')
plt.waitforbuttonpress()


optimizer = torch.optim.SGD(network.parameters(), lr=0.1) #sgd = stochastic gradient descent
lossFunc = torch.nn.MSELoss()
EPOCH = 1000

#train phase
print('='*100)
print('Pytorch MLP')
for ep in range(EPOCH):
    for(input, target) in zip(x, y):
        prediction = network(input)
        loss = lossFunc(prediction, target) #1. nn output, 2.target

        optimizer.zero_grad() #clear gradients for next train
        loss.backward() #backpropagation, compute gradients
        optimizer.step() #apply gradients
    print('epoch ', ep)
    if(ep % 10 == 0):
        prediction = network(x)
        plt.cla()
        plt.axis([-1, 9, -1.5, 1.5])
        plt.scatter(x.data.numpy(), y.data.numpy(), color='blue')
        plt.scatter(x.data.numpy(), prediction.data.numpy(), color='red', alpha=0.7)
        plt.text(6.0, -1.0, 'Epoch = %d' %ep, fontdict={'size': 14, 'color': 'red'})
        plt.text(6.0, -1.25, 'Loss = %.4f' %loss.data.numpy(), fontdict={'size': 14, 'color': 'red'})
        plt.pause(0.01)
    if(loss < 0.0001):
        break

plt.subplots(figsize=(10, 6))
plt.cla()
plt.title('regression analysis - MLP', fontsize=35)
plt.xlabel('independent vatiable', fontsize=24)
plt.ylabel('dependent variable', fontsize=24)
plt.axis([-1, 9, -1.5, 1.5])
plt.scatter(x.data.numpy(), y.data.numpy(), color='blue')
prediction = network(xTest)
plt.plot(xTest.data.numpy(), prediction.data.numpy(), color='red')
plt.savefig('result.png')
plt.show()
plt.waitforbuttonpress()





plt.show()