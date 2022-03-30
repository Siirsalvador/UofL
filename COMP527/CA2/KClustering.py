"""
@Date: 30/03/2022

@Author: Adeola Adebayo

@Description: K-Means and K-Medians clustering implementation on word embeddings (normalised and non-normalised data)
              - COMP 527 CA-2

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from enum import Enum

np.random.seed(35)


class DistanceType(Enum):
    KMEAN = 1
    KMEDIAN = 2


class KClustering:
    def __init__(self, K=4, max_iters=100, distance_type=DistanceType.KMEAN):
        self.n_samples = None
        self.n_features = None
        self.X = None
        self.y = None
        self.original_labels = None
        self.K = K
        self.max_iters = max_iters
        self.distance_type = distance_type
        self.count = None

        # list of sample indices for each cluster
        self.clusters = [[] for _ in range(self.K)]
        # the centers (mean feature vector) for each cluster
        self.centroids = []

    def _get_cluster_labels(self, clusters) -> np.array:
        # each sample will get the label of the cluster it was assigned to
        labels = np.empty(self.n_samples)

        self.pred_labels = dict()
        for cluster_idx, _cluster in enumerate(clusters):
            for sample_index in _cluster:
                self.pred_labels[tuple(self.X[sample_index])] = cluster_idx
                labels[sample_index] = cluster_idx
        return labels

    def _create_clusters(self, centroids) -> list:
        # Assign the samples to the closest centroids to create clusters
        clusters = [[] for _ in range(self.K)]
        for idx, sample in enumerate(self.X):
            centroid_idx = self._closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)
        return clusters

    @staticmethod
    def _closest_centroid(sample, centroids) -> int:
        # distance of the current sample to each centroid
        distances = [euclidean_distance(sample, point) for point in centroids]
        closest_index = np.argmin(distances)
        return closest_index

    def _get_centroids(self, clusters) -> np.array:
        # assign median/mean value of clusters to centroids
        centroids = np.zeros((self.K, self.n_features))
        for cluster_idx, _cluster in enumerate(clusters):
            if self.distance_type == DistanceType.KMEAN:
                cluster_median = np.median(self.X[_cluster], axis=0)
                centroids[cluster_idx] = cluster_median

            if self.distance_type == DistanceType.KMEDIAN:
                cluster_mean = np.mean(self.X[_cluster], axis=0)
                centroids[cluster_idx] = cluster_mean
        return centroids

    def _is_converged(self, centroids_old, centroids) -> bool:
        # distances between each old and new centroids, fol all centroids
        distances = [
            euclidean_distance(centroids_old[i], centroids[i]) for i in range(self.K)
        ]
        return sum(distances) == 0

    def get_cluster_label_count(self) -> None:
        self.count = [[0] * len(np.unique(self.y)) for _ in range(self.K)]
        for cluster_idx, _cluster in enumerate(self.clusters):
            loc = self.count[cluster_idx]
            for sample_index in _cluster:
                loc[self.original_labels[tuple(self.X[sample_index])]] = loc[self.original_labels[
                    tuple(self.X[sample_index])]] + 1

    def predict(self, X, y=None) -> np.array:
        self.X = X
        self.n_samples, self.n_features = X.shape

        if y is not None:
            self.y = y
            self.original_labels = dict()
            for i in range(X.shape[0]):
                self.original_labels[tuple(X[i])] = y[i]

        # initialize
        random_sample_idx = np.random.choice(self.n_samples, self.K, replace=False)
        self.centroids = [self.X[idx] for idx in random_sample_idx]

        # Optimize clusters
        for _ in range(self.max_iters):
            # Assign samples to the closest centroids (create clusters)
            self.clusters = self._create_clusters(self.centroids)

            # Calculate new centroids from the clusters
            centroids_old = self.centroids
            self.centroids = self._get_centroids(self.clusters)
            # print(self.centroids)

            # check if clusters have changed
            if self._is_converged(centroids_old, self.centroids):
                # print("converged at iteration - " + str(_))
                break

        x = self._get_cluster_labels(self.clusters)
        return x

    # noinspection PyTypeChecker
    def get_metrics(self, cluster_metrics) -> tuple[float, float, float]:
        sum_precision = 0
        sum_recall = 0
        sum_fscore = 0

        for idx, item in enumerate(self.count):
            cluster_recall = 0
            cluster_precision = 0
            cluster_fscore = 0

            for _idx, i in enumerate(item):
                if i == 0:
                    continue

                # A(x) across all clusters
                size_i = 0
                for q in range(self.K):
                    h = self.count[q]
                    size_i = size_i + h[_idx]

                # sum of precision for all x in same cluster C(x)
                # A(x) in C(x) / all C(x)
                i_precision = i * (i / sum(item))

                # sum of recall for all x in same cluster C(x)
                # A(x) in C(x) / all A(x)
                i_recall = i * (i / size_i)

                # sum of f_score for all x in same cluster C(x)
                # 2 * (recall(x) * precision(x)) / (recall(x) + precision(x)) 
                i_fscore = i * (2 * ((i / sum(item)) * (i / size_i)) / ((i / size_i) + (i / sum(item))))

                # sum over all instances in cluster
                cluster_recall = cluster_recall + i_recall
                cluster_precision = cluster_precision + i_precision
                cluster_fscore = cluster_fscore + i_fscore

            # average stats over clusters just because
            _k_cluster_metrics = cluster_metrics[self.K]
            _k_cluster_metrics[idx] = tuple(
                [cluster_precision / sum(item), cluster_recall / sum(item), cluster_fscore / sum(item)])

            # sum over entire dataset
            sum_precision = sum_precision + cluster_precision
            sum_recall = sum_recall + cluster_recall
            sum_fscore = sum_fscore + cluster_fscore

        # average over entire dataset
        average_precision = sum_precision / self.X.shape[0]
        average_recall = sum_recall / self.X.shape[0]
        average_fscore = sum_fscore / self.X.shape[0]

        # return as tuple
        return tuple([average_precision, average_recall, average_fscore])


def euclidean_distance(x1, x2) -> float:
    """
    returns euclidean distance between two vectors
    :param x1: vector
    :param x2: vector
    :return: float
    """
    return np.linalg.norm(x1 - x2)


def plot_metrics(k_points, p_points, r_points, f_points, distance_type, norm=False) -> None:
    """
    Plots results

    :param k_points: number of clusters from 1 to k
    :param p_points: average precision for each k
    :param r_points: average recall for each k
    :param f_points: average f-score for each k
    :param distance_type: k-means or k-medians
    :param norm: data normalised or not
    """
    plt.plot(k_points, p_points)
    plt.plot(k_points, r_points)
    plt.plot(k_points, f_points)
    if not norm:
        plt.title("K-Means" if distance_type == DistanceType.KMEAN else "K-Medians")
    else:
        plt.title("K-Means Normalised" if distance_type == DistanceType.KMEAN else "K-Medians Normalised")
    plt.legend(['Precision', 'Recall', 'F-Score'], loc='upper left')
    plt.show()


# noinspection PyTypeChecker
def cluster(X, y, _k=9, distance_type=DistanceType.KMEAN, norm=False) -> None:
    """
    Does clustering and displays results

    :param X: feature vector contains n_samples
    :param y: labels
    :param _k: number of clusters from 1 to k
    :param distance_type: implement k-means or k-medians
    :param norm: data normalised or not
    """
    # contains average precision, recall, and f-score
    # for individual clusters
    cluster_metrics = [[[] for _ in range(j)] for j in range(_k + 1)]

    # contains average precision, recall, and f-score
    # for each k iteration
    k_metrics = [[] for _ in range(_k + 1)]

    for k_value in range(1, _k + 1):
        k_cluster = KClustering(K=k_value, max_iters=150, distance_type=distance_type)
        k_cluster.predict(X, y)
        k_cluster.get_cluster_label_count()
        k_metrics[k_value] = k_cluster.get_metrics(cluster_metrics)

    k_points = np.arange(1, _k + 1)
    p_points = list()
    r_points = list()
    f_points = list()

    for idx, item in enumerate(k_metrics):
        if len(item) > 0:
            p, r, f = item
            p_points.append(p)
            r_points.append(r)
            f_points.append(f)

    print(tuple([k_points, p_points, r_points, f_points, distance_type]))
    # f = open("./tables.txt", "a")
    # f.write(str(tuple([k_points, p_points, r_points, f_points, distance_type])))
    # f.close()
    plot_metrics(k_points, p_points, r_points, f_points, distance_type, norm)


# Testing
if __name__ == "__main__":
    animals = pd.read_csv('./animals', header=None)
    countries = pd.read_csv('./countries', header=None)
    fruits = pd.read_csv('./fruits', header=None)
    veggies = pd.read_csv('./veggies', header=None)

    x_animals = np.array([np.array(str(np.array(i)).split(" "))[1:].astype(float) for i in animals[0]])
    y_animals = np.array([0 for i in animals[0]])

    x_countries = np.array([np.array(str(np.array(i)).split(" "))[1:].astype(float) for i in countries[0]])
    y_countries = np.array([1 for i in countries[0]])

    x_fruits = np.array([np.array(str(np.array(i)).split(" "))[1:].astype(float) for i in fruits[0]])
    y_fruits = np.array([2 for i in fruits[0]])

    x_veggies = np.array([np.array(str(np.array(i)).split(" "))[1:].astype(float) for i in veggies[0]])
    y_veggies = np.array([3 for i in veggies[0]])

    _X = np.concatenate([x_animals, x_countries, x_veggies, x_fruits])
    _y = np.concatenate([y_animals, y_countries, y_fruits, y_veggies])

    k = 9
    cluster(_X, _y, k, DistanceType.KMEAN)
    cluster(_X, _y, k, DistanceType.KMEDIAN)
    m = np.array([[y / np.linalg.norm(x) for y in x] for x in _X])

    cluster(m, _y, k, DistanceType.KMEAN, True)
    cluster(m, _y, k, DistanceType.KMEDIAN, True)
