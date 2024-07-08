"""
functions for GBS clustering
"""
import numpy as np
import itertools

def distance(p1, p2):
        """Euclidean distance of point_1 and point_2"""
        dis = np.sqrt(sum(abs(p1-p2)**2))
        return dis

def construct_adj_mat(data, d_0, dis_func):
    """Construct the adjacent matrix for the given data"""
    num_data = data.shape[-1]
    a = np.zeros([num_data, num_data])
    for i in itertools.combinations(range(num_data), 2):
        dis = dis_func(data[:,i[0]], data[:,i[1]])
        if dis <=d_0:
            a[i] = 1
    return a+a.transpose()