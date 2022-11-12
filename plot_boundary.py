import numpy as np
import matplotlib.pyplot as plt


def predict(inputs , weights):
    output = np.dot(inputs , weights[1:]) + weights[0]   # calc output  
    activation = 1.0 if (output > 0) else 0.0            # threshold
    # if output > 0:
    #     activation = 1
    # else:
    #     activation = 0
    return activation
    

def plot_decision_boundary_m(model):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    step = 0.1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, step),
                         np.arange(y_min, y_max, step))

    points = np.c_[xx.ravel(), yy.ravel()]

    predictedPoints = []
    for point in points:
        predictedPoint = model.predict(point)
        predictedPoints.append(predictedPoint)

    Z = []
    for p in predictedPoints:
        if (np.sum(p) == 1):
            Z.append(np.where(p == 1)[0][0])
        else:
            #Z.append(6)
            Z.append(0)

    Z = np.reshape(Z, xx.shape)

    plt.figure()
    plt.title("Decision Boundary")
    plt.contourf(xx, yy, Z, cmap='gnuplot')
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='gray', cmap='gnuplot')
    plt.show()




def plot_decision_boundary_wm():
    x_min, x_max = point[:, 0].min() - 0.25, point[:, 0].max() + 0.25
    y_min, y_max = point[:, 1].min() - 0.25, point[:, 1].max() + 0.25
    step = 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))     # partition some pieces with step 0.1
    points = np.c_[xx.ravel(), yy.ravel()]
    Z=[]
    for inputs in points:    
        Z.append(predict(inputs, w)) 
    # Z = predict(points, w)
    Z = np.array(Z)
    Z = Z.reshape(xx.shape)
    plt.figure()
    plt.title("Decision Boundary")
    plt.contourf(xx, yy, Z, cmap='Spectral')                # Coloring pieces
    plt.scatter(point[:, 0], point[:, 1], c=label_point, edgecolors = 'gray', cmap='Spectral')
    #plt.cla()
    plt.show()
    plt.pause(0.05)
    plt.waitforbuttonpress()
    plt.close()