import math

def average (x):
    sum = 0.0
    num = 0.0
    for number in x:
        if number != -1:
            sum += number
            num +=1

    return sum / num

def mydist(x, y):
    sum = 0.0
    num = 0.0
    for i, number in enumerate(x):
        if number != -1 and y[i] != -1:
            sum += abs(x[i] - y[i]) **2
            num += 1


    if num == 0.0:
        new_distance = abs(average(x) - average(y))
        #print (x, y, new_distance)
        return new_distance
    avg = math.sqrt(sum)
    #print(x, y, avg)
    return avg

from sklearn.neighbors import NearestNeighbors
import numpy as np
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
nbrs = NearestNeighbors(n_neighbors=3).fit(X)
distances, indices = nbrs.kneighbors(X)
print (distances)
print (np.mean(distances, axis = 1))
