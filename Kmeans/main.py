#################################################
# kmeans: k-means cluster
# Author : zhengyang xu
# Date   : 2020-09-21
#################################################
import numpy as np


def Euclidean_distance(var1, var2):
    squared_dis = 0
    for i in range(len(var1)):
        squared_dis += (var1[i] - var2[i]) ** 2

    return np.sqrt(squared_dis)


def initCentroids(data, k):
    numSamples, dim = data.shape
    centroids = np.zeros(((k, dim)))

    for i in range(k):
        index = int(np.random.uniform(0, numSamples))
        centroids[i] = data.loc[[index]].values[0]

    return centroids


def kmeans(data, k):
    numSamples = data.shape[0]
    centroids = initCentroids(data, k)

    if_change = True
    while if_change:
        # calculate distance from current point to every centroids
        index_list = []
        for i in range(numSamples):
            dis_list = []
            for j in range(k):
                dis_list.append(Euclidean_distance(data.loc[[i]].values[0], centroids[j]))
            index_list.append(np.argsort(dis_list)[0])

        # update cluster centroids
        temp_centroid = centroids
        for centroid in range(k):
            temp_index = [i for i, x in enumerate(index_list) if x == centroid]
            centroids[centroid] = np.array(data.loc[temp_index].sum() / len(temp_index))

        if temp_centroid.all() == centroids.all():
            if_change = False

    return centroids, index_list

