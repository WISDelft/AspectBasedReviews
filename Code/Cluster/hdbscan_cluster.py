from sklearn.datasets.samples_generator import make_blobs
import numpy as np
from sklearn import metrics
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

def print_result(labels, result_array, id):
    #print (id)
    i = 0
    results = []
    for item in result_array:
        if item != None:
            results.append(labels[i])
            i += 1
        else: results.append(None)

    with open('cluster_result.csv', 'a') as f:
                f.write(id + "," + ",".join(str(x) for x in results) + "\n")
    #print (results)

def average_distance(score_array, labels):
    set_labels = set(labels)
    all_points = []
    for cluster in set_labels:
        cluster_points = []
        for i, label in enumerate(labels):
            if label == cluster:
                cluster_points.append(score_array[i])
                #print ("label: " + str(score_array[i]))

        all_points.append(cluster_points)

    num_all = 0.0
    sum_all = 0.0
    for cluster in all_points:
        num = 0.0
        sum = 0.0
        for point1 in cluster:
            for point2 in cluster:
                sum += mydist(point1, point2)
                num += 1
        sum_all += sum/num
        num_all += 1

    return sum_all/num_all

def run_cluster (epislon, score_array, id):
    # blobs, labels = make_blobs(n_samples=2000, n_features=10)
    # print (score_array)
    import hdbscan
    if epislon <= 0: epislon = 0.1
    cluster = hdbscan.HDBSCAN(metric=mydist, min_cluster_size=3, alpha=epislon)  # switch point
    # cluster = hdbscan.HDBSCAN(metric='euclidean')
    cluster.fit(score_array)
    labels = cluster.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    # print(labels)
    # if n_clusters_ > 1:
    #     print("Silhouette Coefficient: %0.3f"
    #           % metrics.silhouette_score(score_array, labels, metric = mydist))

    return cluster
    #print_result(labels, result_array, id)
    print()
def start_cluster(score_array, result_array, id):
    # prev_dis = ''
    # prev_epislon = ''
    # epislon = 0.1
    # distances = []
    # min_slope = 1   #find the elbow in the graph
    # min_cluster = None
    # min_epislon = 0.0
    # while epislon <= 5.0:   #5 is max value
    #     cluster = run_cluster(epislon, score_array, id)
    #     labels = cluster.labels_
    #     n_cluster = len(set(labels)) - (1 if -1 in labels else 0)
    #     distance = average_distance(score_array, labels)
    #     distances.append([distance, epislon])
    #     slope = 1
    #     if (prev_dis != ''):
    #         slope = (distance-prev_dis)/(epislon-prev_epislon)
    #         #print (str(epislon) + ": " + str(slope))
    #     if (slope < min_slope):
    #         min_slope = slope
    #         min_cluster = cluster
    #         min_epislon = epislon
    #
    #     prev_dis = distance
    #     prev_epislon = epislon
    #     #print("Clusters: " + str(n_cluster))
    #     #print("Distance: " + str(distance))
    #     epislon += 0.1
    #
    # print ("Chosen: " + str(min_epislon))
    # print("Cluster: " + str(len(set(min_cluster.labels_)) - (1 if -1 in min_cluster.labels_ else 0)))
    # print_result(min_cluster.labels_, result_array, id)
    # print()
    #plot the choice of epislon and distance
    #distances_np.sort(axis = 0)
    #print (distances_np[:,0])
    #distances_np = np.array(distances)
    from sklearn.neighbors import NearestNeighbors
    print (score_array.shape)
    optimal_eps = 0
    if score_array.shape[0] > 4:
        nbrs = NearestNeighbors(n_neighbors=4).fit(score_array)
        distances, indices = nbrs.kneighbors(score_array)
        means = np.mean(distances, axis=1)
        means.sort()
        max_diff = 0
        for i in range (1, len(means)): #starts from 1 because it takes difference between two points
            if means[i]-means[i-1] > max_diff:
                max_diff = means[i] - means[i-1]
                optimal_eps = means[i-1]
    else: optimal_eps = 5.0
    cluster = run_cluster(optimal_eps, score_array, id)

    print ("Chosen: " + str(optimal_eps))
    print("Cluster: " + str(len(set(cluster.labels_)) - (1 if -1 in cluster.labels_ else 0)))
    print_result(cluster.labels_, result_array, id)
    print()

    # import matplotlib.pyplot as plt
    # plt.plot(means)
    # # plt.xlabel('epislon')
    # plt.ylabel('mean-distance')
    # plt.show()

def clean_array(row):
    #print (row)
    total_None = 0
    for j, item in enumerate(row):
        if j == 0:
            continue
        if 'None' in item:
            #row[j] = 2.5    #switch point
            row[j] = -1
            total_None += 1
        else:
            pass#row[j] = float(item)

    if total_None == 5:
        return None
    return np.array([row[2:7]]).astype(float)

def read_csv():
    with open('cluster_result.csv', 'w') as f:
                f.write("")
    import csv
    with open('score.csv', newline='') as csvfile:
        reader = csv.reader(csvfile)
        csv_header = next(reader)
        score_array = None
        prev_id = ''
        id = ''
        i = 0
        product_num = 0
        #to be deleted
        for row in reader:
            # if product_num > 1:
            #     break

            id = row[0]
            # when the id changes to the next product, print new line and refresh the array
            if prev_id != id:
                if prev_id != '':
                    #all the data under one movie
                    #print (score_array)
                    #print(score_array.shape)
                    print(len(result_array))
                   # print(result_array)
                    start_cluster(score_array, result_array, prev_id)
                    product_num += 1

                    #if product_num >6: break   #to be deleted

                print("start new id!" + id)
                #now refresh the array
                result_array = []
                score_array = None

            prev_id = row[0]
            #print (row)
            if prev_id == id:
                temp = clean_array(row)
                if temp is not None:
                    result_array.append(int(row[1]))
                    if (score_array is not None):
                        score_array = np.concatenate((score_array, temp))
                    else:
                        score_array = temp
                else:
                    result_array.append(None)
            # should happen in the end

            i += 1

read_csv()