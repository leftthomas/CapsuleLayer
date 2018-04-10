import matplotlib as mpl

mpl.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler


def db_scan(data, epsilon=0.2, min_pts=10):
    visited = []
    clusters = []
    for i in range(0, data.shape[0]):
        if data[i] not in np.array(visited):
            visited.append(data[i])
            cluster = []

            for n in range(0, data.shape[0]):
                if LA.norm(data[i] - data[n]) <= epsilon:
                    cluster.append(data[n])

            if len(cluster) >= min_pts:
                print(len(cluster))
                # expanding the cluster
                for point in cluster:
                    if point not in np.array(visited):
                        visited.append(point)
                        for n in range(0, data.shape[0]):
                            if LA.norm(data[n] - point) <= epsilon and data[n] not in np.array(cluster):
                                cluster.append(data[n])
                clusters.append(np.array(cluster))
    return clusters


def my_DBSCAN(data, epsilon=0.2, min_pts=10):
    visited = []
    clusters = []
    for i in range(0, data.shape[0]):
        if data[i] not in np.array(visited):
            visited.append(data[i])
            cluster = []

            for n in range(0, data.shape[0]):
                if LA.norm(data[i] - data[n]) <= epsilon:
                    cluster.append(data[n])

            if len(cluster) >= min_pts:
                print(len(cluster))
                # expanding the cluster
                for point in cluster:
                    if point not in np.array(visited):
                        visited.append(point)
                        for n in range(0, data.shape[0]):
                            if LA.norm(data[n] - point) <= epsilon and data[n] not in np.array(cluster):
                                cluster.append(data[n])
                clusters.append(np.array(cluster))
    return clusters


if __name__ == '__main__':
    # Generate sample data
    centers = [[1, 1], [-1, -1], [1, -1]]
    X, labels_true = make_blobs(n_samples=750, centers=centers, cluster_std=0.4, random_state=0)

    X = StandardScaler().fit_transform(X)

    clusters = my_DBSCAN(X, 0.2, 10)

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 2, 1)
    ax.set_title('Original Data')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.scatter(X[:, 0], X[:, 1])

    ax = fig.add_subplot(1, 2, 2)
    ax.set_title('Clustered Data')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    for cluster in clusters:
        ax.scatter(cluster[:, 0], cluster[:, 1])
        print('center: (' + str(cluster[:, 0].mean()) + ', ' + str(cluster[:, 1].mean()) + ')')

    plt.show()
