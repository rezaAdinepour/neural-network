import numpy as np
import matplotlib.pylab as plt
from nnfs.datasets import spiral_data
import sklearn.datasets


plt.figure(1)
x1, y1= spiral_data(samples=200, classes=3)
plt.scatter(x1[:, 0], x1[:, 1], c=y1, cmap='brg')



x, y = sklearn.datasets.make_blobs(n_samples=500, n_features=2, centers=[(1, -1), (2, 2)], cluster_std=0.6, center_box=(-10.0, 10.0),
                                   shuffle=True, random_state=None, return_centers=False)
plt.figure(2)
plt.scatter(x[:, 0], x[:, 1], c=y, cmap='brg')


plt.show()