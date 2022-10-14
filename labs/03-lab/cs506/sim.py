import numpy as np

def euclidean_dist(x, y):
    res = 0
    for i in range(len(x)):
        res += (x[i] - y[i])**2
    return res**(1/2)

def manhattan_dist(x, y):
    xy = []
    for xi, yi in zip(x, y):
        xy.append(abs(xi-yi))
    return sum(xy)

def jaccard_dist(x, y):
    intx = abs(len(x.intersection(y)))
    unn = abs(len(x.union(y)))
    return 1 - intx/unn

def cosine_sim(x, y):
    return np.dot(x,y) /(np.absolute * np.absolute(y))

# Feel free to add more
