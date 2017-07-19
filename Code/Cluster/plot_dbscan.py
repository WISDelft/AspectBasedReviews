# -*- coding: utf-8 -*-
"""
===================================
Demo of DBSCAN clustering algorithm
===================================

Finds core samples of high density and expands clusters from them.

"""
print(__doc__)

import numpy as np

from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler



result_array = []
##############################################################################
# Plot result
def plot(X, labels, core_samples_mask, n_clusters_):
    import matplotlib.pyplot as plt

    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = 'k'

        class_member_mask = (labels == k)

        xy = X[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=14)

        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=6)

    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()

##############################################################################
def average (x):
    sum = 0.0
    num = 0.0
    for number in x:
        if number != -1:
            sum += number
            num +=1

    return sum/num
# Compute DBSCAN
def mydist(x, y):
    sum = 0.0
    num = 0.0
    for i, number in enumerate(x):
        if number != -1 and y[i] != -1:
            sum += abs(x[i] - y[i])
            num += 1


    if num == 0.0:
        new_distance = abs(average(x) - average(y))
        #print (x, y, new_distance)
        return new_distance

    avg = sum/num
    #print(x, y, avg)
    return avg

def dbscan(X):
    global result_array
    db = DBSCAN(eps=0.25, min_samples=3, metric = mydist).fit(X)
    labels = db.labels_
    # core_samples_mask = np.zeros_like(labels, dtype=bool)
    # core_samples_mask[db.core_sample_indices_] = True

    #label all the reviews
    label_array = []
    label_num = 0
    for item in result_array:
        if item != None:
            if labels[label_num] != -1: label_array.append(labels[label_num])
            label_num += 1
        else:
            label_array.append(None)
    print(label_array)

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    print('number of clusters: %d' % n_clusters_)
    # print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
    # print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
    # print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
    # print("Adjusted Rand Index: %0.3f"
    #       % metrics.adjusted_rand_score(labels_true, labels))
    # print("Adjusted Mutual Information: %0.3f"
    #       % metrics.adjusted_mutual_info_score(labels_true, labels))

    if n_clusters_ > 1:
        print("Silhouette Coefficient: %0.3f"
              % metrics.silhouette_score(X, labels, metric = mydist))

    #plot(X, labels, core_samples_mask, n_clusters_)



##############################################################################
# Generate sample data
def start_dbscan():
    centers = [[1, 1], [-1, -1], [1, -1]]
    X, labels_true = make_blobs(n_samples=750, centers=centers, cluster_std=0.4,
                                random_state=0)
    #print (type(X))
    #print (X.shape)
    X = StandardScaler().fit_transform(X)
    dbscan(X)



def clean_array(row):
    #print (row)
    total_None = 0
    for j, item in enumerate(row):
        if j == 0:
            continue
        if 'None' in item:
            row[j] = -1
            total_None += 1
        else:
            pass#row[j] = float(item)

    if total_None == 5:
        return None
    return np.array([row[2:]]).astype(float)

def read_csv():
    global result_array
    import csv
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
            # when the id changes to the next product, print new line and refresh the array
            if prev_id != id:
                if prev_id != '':
                    #all the data under one movie
                    #print (score_array)
                    print(score_array.shape)
                    print(len(result_array))
                    print(result_array)
                    dbscan(score_array)

                    break   #to be deleted

                print("start new id!" + id)
                print ()
                #now refresh the array
                result_array = []
                score_array = None

            prev_id = row[0]
            #print (row)
            if prev_id == id:
                temp = clean_array(row)
                if temp != None:
                    result_array.append(int(row[1]))
                    if (score_array != None):
                        score_array = np.concatenate((score_array, temp))
                    else:
                        score_array = temp
                else:
                    result_array.append(None)
            # should happen in the end

            i += 1





read_csv()
