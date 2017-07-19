"""
=======================================
Clustering text documents using k-means
=======================================

This is an example showing how the scikit-learn can be used to cluster
documents by topics using a bag-of-words approach. This example uses
a scipy.sparse matrix to store the features instead of standard numpy arrays.

Two feature extraction methods can be used in this example:

  - TfidfVectorizer uses a in-memory vocabulary (a python dict) to map the most
    frequent words to features indices and hence compute a word occurrence
    frequency (sparse) matrix. The word frequencies are then reweighted using
    the Inverse Document Frequency (IDF) vector collected feature-wise over
    the corpus.

  - HashingVectorizer hashes word occurrences to a fixed dimensional space,
    possibly with collisions. The word count vectors are then normalized to
    each have l2-norm equal to one (projected to the euclidean unit-ball) which
    seems to be important for k-means to work in high dimensional space.

    HashingVectorizer does not provide IDF weighting as this is a stateless
    model (the fit method does nothing). When IDF weighting is needed it can
    be added by pipelining its output to a TfidfTransformer instance.

Two algorithms are demoed: ordinary k-means and its more scalable cousin
minibatch k-means.

Additionally, latent semantic analysis can also be used to reduce dimensionality
and discover latent patterns in the data. 

It can be noted that k-means (and minibatch k-means) are very sensitive to
feature scaling and that in this case the IDF weighting helps improve the
quality of the clustering by quite a lot as measured against the "ground truth"
provided by the class label assignments of the 20 newsgroups dataset.

This improvement is not visible in the Silhouette Coefficient which is small
for both as this measure seem to suffer from the phenomenon called
"Concentration of Measure" or "Curse of Dimensionality" for high dimensional
datasets such as text data. Other measures such as V-measure and Adjusted Rand
Index are information theoretic based evaluation scores: as they are only based
on cluster assignments rather than distances, hence not affected by the curse
of dimensionality.

Note: as k-means is optimizing a non-convex objective function, it will likely
end up in a local optimum. Several runs with independent random init might be
necessary to get a good convergence.

"""

# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#         Lars Buitinck
# License: BSD 3 clause

from __future__ import print_function

from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn import metrics

from sklearn.cluster import KMeans, MiniBatchKMeans

import logging
from optparse import OptionParser
import sys
from time import time

import numpy as np


def start_cluster(review_list, true_k, filename):
    # Display progress logs on stdout
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')

    # parse commandline arguments
    op = OptionParser()
    op.add_option("--lsa",
                  dest="n_components", type="int",
                  help="Preprocess documents with latent semantic analysis.")
    op.add_option("--no-minibatch",
                  action="store_false", dest="minibatch", default=True,
                  help="Use ordinary k-means algorithm (in batch mode).")
    op.add_option("--no-idf",
                  action="store_false", dest="use_idf", default=True,
                  help="Disable Inverse Document Frequency feature weighting.")
    op.add_option("--use-hashing",
                  action="store_true", default=False,
                  help="Use a hashing feature vectorizer")
    op.add_option("--n-features", type=int, default=10000,
                  help="Maximum number of features (dimensions)"
                       " to extract from text.")
    op.add_option("--verbose",
                  action="store_true", dest="verbose", default=False,
                  help="Print progress reports inside k-means algorithm.")

    #print(__doc__)
    #op.print_help()

    (opts, args) = op.parse_args()
    if len(args) > 0:
        op.error("this script takes no arguments.")
        sys.exit(1)


    ###############################################################################
    # Load some categories from the training set
    categories = [
        'alt.atheism',
        'talk.religion.misc',
        'comp.graphics',
        'sci.space',
    ]
    # Uncomment the following to do the analysis on all the categories
    #categories = None

    #print("Loading 20 newsgroups dataset for categories:")
    #print(categories)

    dataset = fetch_20newsgroups(subset='all', categories=categories,
                                 shuffle=True, random_state=42)
    # print (type(dataset.data))
    # print("%d documents" % len(dataset.data))
    # print("%d categories" % len(dataset.target_names))
    print()

    labels = dataset.target

    #print("Extracting features from the training dataset using a sparse vectorizer")
    # t0 = time()
    # if opts.use_hashing:
    #     if opts.use_idf:
    #         # Perform an IDF normalization on the output of HashingVectorizer
    #         hasher = HashingVectorizer(n_features=opts.n_features,
    #                                    stop_words='english', non_negative=True,
    #                                    norm=None, binary=False)
    #         vectorizer = make_pipeline(hasher, TfidfTransformer())
    #     else:
    #         vectorizer = HashingVectorizer(n_features=opts.n_features,
    #                                        stop_words='english',
    #                                        non_negative=False, norm='l2',
    #                                        binary=False)
    # else:
    #     vectorizer = TfidfVectorizer(max_df=0.5, max_features=opts.n_features,
    #                                  min_df=2, stop_words='english',
    #                                  use_idf=opts.use_idf)


    vectorizer = TfidfVectorizer(max_df=0.8, max_features=200000,
                                       min_df=0, stop_words='english',
                                       use_idf=True, tokenizer=tokenize_and_stem, ngram_range=(1, 1))   #gram is the window

    #X = vectorizer.fit_transform(dataset.data)
    X = vectorizer.fit_transform(review_list)
    print (X)
    print (vectorizer)

    #print("done in %fs" % (time() - t0))
    #print("n_samples: %d, n_features: %d" % X.shape)
    print()

    if opts.n_components:
        print("Performing dimensionality reduction using LSA")
        t0 = time()
        # Vectorizer results are normalized, which makes KMeans behave as
        # spherical k-means for better results. Since LSA/SVD results are
        # not normalized, we have to redo the normalization.
        svd = TruncatedSVD(opts.n_components)
        normalizer = Normalizer(copy=False)
        lsa = make_pipeline(svd, normalizer)

        X = lsa.fit_transform(X)

        print("done in %fs" % (time() - t0))

        explained_variance = svd.explained_variance_ratio_.sum()
        print("Explained variance of the SVD step: {}%".format(
            int(explained_variance * 100)))

        print()


    ###############################################################################
    # Do the actual clustering

    # if opts.minibatch:
    #     km = MiniBatchKMeans(n_clusters=true_k, init='k-means++', n_init=1,
    #                          init_size=1000, batch_size=1000, verbose=opts.verbose)
    # else:
    km = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1, algorithm= 'full',
                verbose=opts.verbose)

    #print("Clustering sparse data with %s" % km)
    t0 = time()
    km.fit(X)
    #print("done in %0.3fs" % (time() - t0))
    print()

    # print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels, km.labels_))
    # print("Completeness: %0.3f" % metrics.completeness_score(labels, km.labels_))
    # print("V-measure: %0.3f" % metrics.v_measure_score(labels, km.labels_))
    # print("Adjusted Rand-Index: %.3f"
    #       % metrics.adjusted_rand_score(labels, km.labels_))
    print (len(review_list))
    silhouette_avg = metrics.silhouette_score(X, km.labels_, sample_size=len(review_list))
    print("Silhouette Coefficient: %0.3f"
          % silhouette_avg)
    with open(filename, 'a') as f:
        f.write("Silhouette score: " + str(silhouette_avg) + "\n")
    #
    # print()


    if not opts.use_hashing:
        print("Top terms per cluster:")

        if opts.n_components:
            original_space_centroids = svd.inverse_transform(km.cluster_centers_)
            order_centroids = original_space_centroids.argsort()[:, ::-1]
        else:
            order_centroids = km.cluster_centers_.argsort()[:, ::-1]

        terms = vectorizer.get_feature_names()
        print (terms)
        for i in range(true_k):
            print("Cluster %d:" % i, end='')
            with open(filename, 'a') as f:
                f.write("Cluster "+str(i)+":  ")
            for ind in order_centroids[i, :10]:
                print(' %s' % terms[ind], end='')
                with open(filename, 'a') as f:
                    f.write(terms[ind]+" ")

            with open(filename, 'a') as f:
                f.write( "\n")
            print()
            print("documents in this cluster: ")
            limit = 0;
            for j in range(0, len(km.labels_)):
                if km.labels_[j] == i:
                    print(km.labels_[j], review_list[j])
                    with open(filename, 'a') as f:
                        f.write(review_list[j] + "\n")
                    limit += 1
                if limit >= 15:
                    break
            print()

def tokenize_and_stem(text):
    import nltk
    import re
    from nltk.stem.snowball import SnowballStemmer
    from nltk.stem.wordnet import WordNetLemmatizer
    import string
    # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
    #stemmer = SnowballStemmer("english")
    stemmer = WordNetLemmatizer()
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))

    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent) if word not in stop_words]
    filtered_tokens = []
    # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
    wierd_words = ["s", "nt"]
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token.translate(str.maketrans('','',string.punctuation)))
    stems = [stemmer.lemmatize(t) for t in filtered_tokens if t not in wierd_words]
    return stems

