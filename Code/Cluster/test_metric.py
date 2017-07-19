from sklearn.cluster import KMeans
import numpy as np

def mydist(x, y):
    return np.sum((x-y)**2)
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])

kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
print (kmeans.labels_)