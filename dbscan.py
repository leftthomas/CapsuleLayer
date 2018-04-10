import matplotlib as mpl
import torch

mpl.use('TkAgg')
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler


def db_scan(data, epsilon=0.2, min_pts=10):
    visited = []
    clusters = []
    for i in range(0, data.size(0)):
        if not any(data[i] is e or data[i].equal(e) for e in visited):
            visited.append(data[i])
            cluster = []

            for n in range(0, data.size(0)):
                if torch.norm(data[i] - data[n]) <= epsilon:
                    cluster.append(data[n])

            if len(cluster) >= min_pts:
                # expanding the cluster
                for point in cluster:
                    if not any(point is e or point.equal(e) for e in visited):
                        visited.append(point)
                        for n in range(0, data.size(0)):
                            if torch.norm(data[n] - point) <= epsilon and not any(
                                    data[n] is e or data[n].equal(e) for e in cluster):
                                cluster.append(data[n])
                for index in range(len(cluster)):
                    cluster[index] = cluster[index].numpy()
                clusters.append(np.array(cluster))
    return clusters


def test_db_scan(data, epsilon=0.2, min_pts=10):
    clusters = []
    for batch in range(data.size(0)):
        visited_node_mask = torch.zeros(data.size(1))
        cluster_centers = []
        for i in range(data.size(1)):
            if visited_node_mask[i] == 0:
                visited_node_mask[i] = 1
                cluster_indexes = (torch.norm(data[batch] - data[batch][i], p=2, dim=-1) <= epsilon).nonzero().view(
                    -1).tolist()

                if len(cluster_indexes) >= min_pts:
                    # expanding the cluster_indexes
                    for j in cluster_indexes:
                        if visited_node_mask[j] == 0:
                            visited_node_mask[j] = 1

                            expanded_indexes = torch.norm(data[batch] - data[batch][j], p=2, dim=-1) <= epsilon
                            # remove already existed indexes
                            expanded_indexes[cluster_indexes] = 0
                            cluster_indexes += expanded_indexes.nonzero().view(-1).tolist()
                    cluster_center = torch.index_select(data[batch], dim=0, index=torch.LongTensor(cluster_indexes)) \
                        .mean(dim=0, keepdim=True)
                    cluster_centers.append(cluster_center)
        clusters.append(torch.cat(cluster_centers, dim=0).unsqueeze(dim=0))
    return torch.cat(clusters, dim=0)


if __name__ == '__main__':

    # Generate sample data
    centers1 = [[1, 1], [-1, -1], [1, -1]]
    centers2 = [[1, 1], [-1, -1], [1, -1], [-1, 1]]
    X1, _ = make_blobs(n_samples=750, centers=centers1, cluster_std=0.4, random_state=0)
    X2, _ = make_blobs(n_samples=750, centers=centers2, cluster_std=0.4, random_state=0)
    X1 = StandardScaler().fit_transform(X1)
    X2 = StandardScaler().fit_transform(X2)
    X1 = torch.from_numpy(X1)
    X2 = torch.from_numpy(X2)
    clusters = db_scan(X1, 0.2, 10)

    # fig = plt.figure(figsize=(10, 5))
    # ax = fig.add_subplot(1, 2, 1)
    # ax.set_title('Original Data')
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.scatter(X[:, 0], X[:, 1])
    #
    # ax = fig.add_subplot(1, 2, 2)
    # ax.set_title('Clustered Data')
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    for cluster in clusters:
        # ax.scatter(cluster[:, 0], cluster[:, 1])
        print('center: (' + str(cluster[:, 0].mean()) + ', ' + str(cluster[:, 1].mean()) + ')')

    # plt.show()
    clusters = test_db_scan(torch.cat((X1.unsqueeze(dim=0), X2.unsqueeze(dim=0)), dim=0), 0.2, 10)
    print(clusters)
