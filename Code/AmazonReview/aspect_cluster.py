from sklearn import cluster, datasets
import numpy as np

def start_cluster(filename, aspect_array, review_person, k):
    #print(X_iris.shape)

    ##############np array  concatenate example##############
    # test = np.array([[2,3,4,5]])
    # new = np.array([[1, 2, 3, 4]])
    # print(new.shape)
    # add = np.concatenate((test, new))
    # print(add)

    k_means = cluster.KMeans(n_clusters=k, algorithm= "full")
    print (aspect_array)
    print(aspect_array.shape)
    k_means.fit(aspect_array)


    print (aspect_array.shape, len(k_means.labels_))
    from sklearn import metrics
    from sklearn.metrics import silhouette_samples, silhouette_score
    silhouette_avg = metrics.silhouette_score(aspect_array, k_means.labels_,
                                      metric='euclidean',
                                      sample_size=int(len(review_person)))
    print()
    with open(filename, 'a') as f:
        f.write("For n_clusters ="+str(k)+
          "The average silhouette_score is :"+str(silhouette_avg)  )
        f.write("\n")

    results = k_means.labels_

    for i in range(0, k):
        print ("Cluster ", i, ": ", k_means.cluster_centers_[i])
        with open(filename, 'a') as f:
            f.write("Cluster "+str(i)+": "+str(k_means.cluster_centers_[i]))
            f.write("\n")
        limit = 0
        for j, value in enumerate(results):
            if value == i:
                print (review_person[j])
                with open(filename, 'a') as f:
                    f.write(review_person[j])
                    f.write("\n")
                limit += 1
            if limit >= 15:
                break

#start_cluster(np.array([[]]))