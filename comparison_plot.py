import time

import numpy as np
import matplotlib.pyplot as plt

from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice

from DBSCAN import DBSCAN

np.random.seed(0)

# ============
# Generate datasets. We choose the size big enough to see the scalability
# of the algorithms, but not too big to avoid too long running times
# ============
n_samples = 100
noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5,
                                      noise=.05)
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)
blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
no_structure = np.random.rand(n_samples, 2), None

# Anisotropicly distributed data
random_state = 170
X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
X_aniso = np.dot(X, transformation)
aniso = (X_aniso, y)

# blobs with varied variances
varied = datasets.make_blobs(n_samples=n_samples,
                             cluster_std=[1.0, 2.5, 0.5],
                             random_state=random_state)

# ============
# Set up cluster parameters
# ============
plt.figure(figsize=(10, 10))
plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,
                    hspace=.01)


default_epsilon = 0.3

varied_eps = 0.18
aniso_eps = 0.15

names = ["noisy_circles", "noisy_moons", "varied", "aniso", "blobs", "no_structure"]
datasets = [noisy_circles, noisy_moons, varied, aniso, blobs, no_structure]

plot_num = 1
i=0

for dataset in datasets:

    print(names[i])
    i += 1

    X = dataset[0]

    X = StandardScaler().fit_transform(X)

    if dataset is varied:
        eps = default_epsilon
    elif dataset is aniso:
        eps = aniso_eps
    else:
        eps = default_epsilon


### My DBSCAN

    model = DBSCAN(eps)

    t0 = time.time()

    model.fit(X)

    t1 = time.time()


    y_pred = model.cluster.labels

    colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                                '#f781bf', '#a65628', '#984ea3',
                                                '#999999', '#e41a1c', '#dede00']),
                                        int(max(y_pred) + 1))))
    
    colors = np.append(colors, ["#000000"])

    plt.subplot(len(datasets), 2, plot_num)
    if i < 2:
        plt.title("Mine", size = 18)

    plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_pred])
    plt.xlim(-2.5, 2.5)
    plt.ylim(-2.5, 2.5)
    plt.xticks(())
    plt.yticks(())
    plt.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),
                transform=plt.gca().transAxes, size=15,
                horizontalalignment='right')
    
    plot_num += 1


    ### SKlearn DBSCAN

    dbscan = cluster.DBSCAN(eps = eps)

    t0 = time.time()

    dbscan.fit(X)

    t1 = time.time()

    y_pred = dbscan.labels_.astype(np.int)

    colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                                '#f781bf', '#a65628', '#984ea3',
                                                '#999999', '#e41a1c', '#dede00']),
                                        int(max(y_pred) + 1))))

    colors = np.append(colors, ["#000000"])

    plt.subplot(len(datasets), 2, plot_num)

    if i < 2:
        plt.title("Sklearn", size = 18)

    plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_pred])

    plt.xlim(-2.5, 2.5)
    plt.ylim(-2.5, 2.5)
    plt.xticks(())
    plt.yticks(())
    plt.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),
                transform=plt.gca().transAxes, size=15,
                horizontalalignment='right')

    plot_num += 1


    

plt.show()


