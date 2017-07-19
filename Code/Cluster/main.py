import csv
import numpy as np

def clean_array(row):
    for j, item in enumerate(row):
        if j == 0:
            continue
        if item == ' None':
            row[j] = np.nan
        else:
            pass#row[j] = float(item)
    return np.array([row[1:]]).astype(float)


import numpy as np
from sklearn.cluster import KMeans

def kmeans_missing(X, n_clusters, max_iter=2000):
    """Perform K-Means clustering on data with missing values.

    Args:
      X: An [n_samples, n_features] array of data to cluster.
      n_clusters: Number of clusters to form.
      max_iter: Maximum number of EM iterations to perform.

    Returns:
      labels: An [n_samples] vector of integer labels.
      centroids: An [n_clusters, n_features] array of cluster centroids.
      X_hat: Copy of X with the missing values filled in.
    """

    # Initialize missing values to their column means

    #print (X)
    missing = ~np.isnan(X)
    mu = np.nanmean(X, 0, keepdims=1)
    mu = np.where(np.isnan(mu),3.0, mu)
    #print (mu)
    X_hat = np.where(missing, X, mu)
    #print (X_hat)

    for i in range(max_iter):
        # if i > 0:
        #     # initialize KMeans with the previous set of centroids. this is much
        #     # faster and makes it easier to check convergence (since labels
        #     # won't be permuted on every iteration), but might be more prone to
        #     # getting stuck in local minima.
        #     k_means = KMeans(n_clusters, init=prev_centroids)
        # else:
        #     # do multiple random initializations in parallel
        #     k_means = KMeans(n_clusters, n_jobs=-1)

        k_means = KMeans(n_clusters)

        # perform clustering on the filled-in data
        labels = k_means.fit_predict(X_hat)
        centroids = k_means.cluster_centers_

        # fill in the missing values based on their cluster centroids
        X_hat[missing] = centroids[labels][missing]

        # when the labels have stopped changing then we have converged
        if i > 0 and np.all(labels == prev_labels):
            break

        prev_labels = labels
        prev_centroids = k_means.cluster_centers_

    from sklearn import metrics
    silhouette_avg = metrics.silhouette_score(X_hat, k_means.labels_,
                                              metric='euclidean',
                                              sample_size=int(X_hat.shape[0]))
    #print (silhouette_avg)
    #print (labels)
    return silhouette_avg,labels

def find_best_k(array):
    max_score = -1
    best_k = 1  #if k equals to 1, then something's wrong
    best_labels = ''
    for k in range (2,11):
        if k >= array.shape[0]:
            break
        (score, labels) = kmeans_missing(array, k)
        if score > max_score:
            max_score = score
            best_k = k
            best_labels = labels

    return best_k, best_labels

filename = "cluster.csv"
with open(filename, 'w') as f:
    f.write("id,best_k,cluster_labels \n")

with open('score.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    csv_header = next(reader)
    score_array = None
    prev_id = ''
    id = ''
    i = 0
    for row in reader:
        if i > 5000:
            break

        id = row[0]
        if prev_id != id:
            if prev_id != '':
                print("start new id!" + prev_id)
                (best_k, best_labels) = find_best_k(score_array)
                print (best_labels)
                print ("Best k: " + str(best_k))
                line = prev_id + "," + str(best_k) +"," + " ".join(str(m) for m in best_labels) + "\n"
                with open(filename, 'a') as f:
                    f.write(line)
                print ()
            #now refresh the array
            score_array = clean_array(row)
        if prev_id == '':
            prev_id = row[0]



        temp = clean_array(row)
        score_array = np.concatenate((score_array, temp))
        #print(row)
        #should happen in the end
        prev_id = row[0]
        i += 1
# print (score_array.shape)
# gen = np.random.RandomState(None)
# missing = gen.rand(*score_array.shape) < 0.3
# print (missing)