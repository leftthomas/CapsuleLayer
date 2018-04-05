import matplotlib as mpl

mpl.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler


def my_DBSCAN(data, epsilon, minPts):
    # epsilon is the neighbourhood distance
    # minPts is the  minimum number of nodes required to form a cluster
    # array to keep track of visited nodes
    visited = []
    clusters = {}
    cluster_num = 0
    # for each node in database
    for i in range(0, data.shape[0]):
        if data[i] not in np.array(visited):
            visited.append(data[i])
            cluster = []

            # now we need to calculate the distance between this ith node and all other nodes
            for n in range(0, data.shape[0]):
                # if distance is less than epsilon then we add this node to cluster
                if LA.norm(data[i] - data[n]) <= epsilon:
                    cluster.append(data[n])

            # if length of cluster is greater than minPts, then add it to clusters
            if len(cluster) >= minPts:
                print(len(cluster))
                print("Yes! Lets form a cluster")
                # expanding the cluster
                for point in cluster:
                    if point not in np.array(visited):
                        visited.append(point)
                        for n in range(0, data.shape[0]):
                            # if distance between point and this nth node is less than epsilon
                            if LA.norm(data[n] - point) <= epsilon and data[n] not in np.array(cluster):
                                cluster.append(data[n])
                clusters[cluster_num] = cluster
                cluster_num += 1

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(1, 2, 1)
    ax.set_title('Original Data')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.scatter(data[:, 0], data[:, 1])

    ax = fig.add_subplot(1, 2, 2)
    ax.set_title('Clustered Data')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    for key in clusters.keys():
        ax.scatter(np.array(clusters[key])[:, 0], np.array(clusters[key])[:, 1])

    plt.show()
    return clusters


if __name__ == '__main__':
    # Generate sample data
    centers = [[1, 1], [-1, -1], [1, -1]]
    X, labels_true = make_blobs(n_samples=750, centers=centers, cluster_std=0.4, random_state=0)

    X = StandardScaler().fit_transform(X)

    c = my_DBSCAN(X, 0.2, 10)
