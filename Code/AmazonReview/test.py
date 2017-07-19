from sklearn.metrics.pairwise import paired_distances
X = [[1,2,3,4]]
Y = [[4,5,6,7]]
print(paired_distances(X, Y))