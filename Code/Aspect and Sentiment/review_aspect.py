import pandas as pd
import gzip, re, nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet as wn
from nltk.app.wordnet_app import SIMILAR
from numpy.distutils import numpy_distribution
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords
from gensim import corpora, models, similarities
import gensim
from scipy.signal.ltisys import lsim
import operator
from nltk.tokenize import RegexpTokenizer
import Similarity as similarity
from macpath import join
import string
import numpy as np

def parse(path):
  g = gzip.open(path, 'r')
  for l in g:
    yield eval(l)

def stem_person(word):
    if word[-2:].lower() in ["or" , "er"]:
        return word[:-2]
    else:
        return word
# use similarity to find aspect related words to count score


#repleace non-alphabet characters

#clean_words = clean_review.split(" ")
#Stemmer, choose words related to act aspect in the movie


    

#f = open("compare.txt","w")


# print get_similarity(wn.synsets("act"), wn.synsets(stem_person("actor")))
# print get_similarity(wn.synsets("act"), wn.synsets(stem_person("actress")))
# print get_similarity(wn.synsets("act"), wn.synsets(stem_person("role")))
# print get_similarity(wn.synsets("act"), wn.synsets(stem_person("cast")))
# print get_similarity(wn.synsets("act"), wn.synsets(stem_person("performance")))
# print get_similarity(wn.synsets("act"), wn.synsets(stem_person("play")))

def aspect_rank():
    type = "avg"
    top_20_words = ["horror","way","year","isnt",
                    "man","comedy","director","role","performance",
                    "didnt","life","plot","cast","star","time","love","actor","story","character","scene"]
    top_20_score = [0.0659288553863007,0.0666518054740113,0.0670676855595146,
                    0.0677858589109134,0.0713570077473513,0.0745651678541936,
                    0.0768834166829233,0.0778697784906768,0.0815467375831749,
                    0.0829649663600186,0.0855723474069187,0.0857610020090583,
                    0.0902457257173451,0.0959142221391288,0.127113291972874,
                    0.129745817753784,0.144022588383893,0.176694803214571,
                    0.17987431763156,0.218888637719266]
    aspect_rank = {}
    for i, word in enumerate(top_20_words):
        score = similarity.get_similarity(wn.synsets(word), wn.synsets("movie"), type)
        #print (word, score)
        aspect_rank[word] = score +top_20_score[i] * 3 #weight, 1:3

    sorted_words = sorted(aspect_rank.items(), key=operator.itemgetter(1), reverse = True)
    for word in sorted_words:
        print (word)

def start_analyze():
    lmtzr = WordNetLemmatizer() #words are still readable by doing this
    stemmer = SnowballStemmer("english")
    all_review = ""    
    reviewer_id = {}
    prev_product = ''
    product_id = ''
    product_num = 0     #the number of products, starts from 1
    act_sets = wn.synsets("act")
    num_review = 0    #for counting number of reviews for each product
    review_list = []
    review_person = []
    num_aspect = 0
    review_each_movie = []
    tokenizer = RegexpTokenizer(r'\w+')
    stop_words = set(stopwords.words('english'))
    nouns_verbs_dic = {}
    corpus = []  #all the words appeared in reviews, no duplicates
    possible_words = []
    ratings = []
    has_found = False
    #word2vector
    # import word_vector
    # model = word_vector.start_word_vector()

    # with open('review_score.txt', 'w') as f:
    #             f.write("id,,, review_id2,,, review,,, acting,,, directing,,, scene,,, character,,, story \n")
    pos_num = 0
    neg_num = 0
    sum_balance = 0
    num_balance = 0
    total = 0
    dataset = "reviews_Movies_and_TV_5.json.gz"
   # dataset = "reviews_Books_5.json.gz"
    for i, review in enumerate(parse(dataset)):
        product_id = review['asin']
        # if num_review > 5:
        #     if product_id == prev_product: continue;

        #print(str(i) + " " + product_id + " ")
        ############# for looking for the product #################
        # if product_id == '0780630904':
        #     has_found = True
        #     continue
        # if not has_found: continue
        ############# for looking for the product #################
        #
        # wanted_product = '0767806239' #'0767802454'
        # if (product_id == wanted_product):
        # #if (product_id == '0767802454'):
        #     overall_rate = float(review['overall'])
        #     #print(overall_rate)
        #     total += 1
        #     if overall_rate >= 4: pos_num += 1;
        #     if overall_rate < 3: neg_num += 1;

        # if (product_id != prev_product):
        #     if prev_product == wanted_product:
        #         print (total, pos_num, neg_num)
        #         print (pos_num/neg_num)
            #####################try simple clustering#####################
            # if len(review_person) > 0 and len(review_person) > k:
            #     import clustering
            #     clustering.start_cluster(review_person, k, filename)
            ##########perform aspect clustering################
            # if prev_product != '':
            #     print("*******start clustering***********")
            #
            #     import aspect_cluster as cluster
            #     print(score_array.shape, len(review_person))
            #     if len(review_person) > k:
            #         cluster.start_cluster(filename, score_array, review_person, k)
            #     score_array = np.array([similarity.aspect_included(num_review, prev_product, this_review, 0.471, "minmax")])  # 0.481 for avg

        #####################print product number#####################
            # with open('score.csv', 'a') as f:
            #     f.write("*" * 30+str(product_num) + "th product" +product_id +"*" * 30)
            #     f.write("\n")
        #Select one movie only
        #print review['reviewText']

        #Choose a specific product number, first 0005019281, second product 0005119367,

        this_review = review['reviewText']
        #############trying word2vec######################
        # characters = [letter.lower() if letter.isalpha() else ' ' for letter in this_review]
        # sentence = "".join(characters)
        # tag_sentence = nltk.pos_tag([lmtzr.lemmatize(word) for word in sentence.split()])
        # print(this_review)
        # for i, word in enumerate(sentence.split()):
        #     word = lmtzr.lemmatize(word)
        #     #print (word + " " + tag_sentence[i][1])
        #     aspect= lmtzr.lemmatize('acting')
        #
        #     if "NN" in tag_sentence[i][1] and len(word) > 1 :
        #         try:
        #             score = model.similarity(word, aspect)
        #         except:
        #             continue
        #         if score > 0.86:
        #             num_aspect += 1
        #             print(word+" " + aspect + " " + str(score))
        #
        # print()
        if (product_id == prev_product):
            num_review += 1#         continue;
            ratings.append(review['overall'])
            ################analyze each review which aspects are inluded#####################
        #if (prev_product) == '':        #initialize the value
            # score_array = np.array([similarity.aspect_included(num_review, product_id, this_review, 0.471, "minmax")])  # 0.481 for avg



        #if (product_id == prev_product):
            # aspect_score = np.array([similarity.aspect_included(num_review, product_id, this_review, 0.471, "minmax")])   # 0.481 for avg
            # score_array = np.concatenate((score_array, np.array(aspect_score)))
        #print(score_array)
        #print(score_array.shape)



        if prev_product == '': #initialte the first one

            product_num += 1
            prev_product = product_id
            all_review += review['reviewText']
            all_review += " \n"
            review_person.append(review['reviewText'])
            #print prev_product, product_id
        elif product_id == prev_product:    #same product? then add reviews together
            all_review += review['reviewText']
            all_review += " \n"
            review_person.append(review['reviewText'])
            # overall_rate = float(review['overall'])
            # #print(overall_rate)
            # if overall_rate >= 3.5: pos_num += 1;
            # if overall_rate <= 3: neg_num += 1;

            #print prev_product, product_id
        elif product_id != prev_product: #product_id no equals to previous product

            #print ("number: " + str(num_aspect))
            #print prev_product, product_id
            #print prev_product, product_id
            #print all_review
            #review_person.append(review['reviewText'])


            print("*" * 30, product_num, "th product", prev_product  , " in total: ", str(i),"*" * 30)


            #print(str(pos_num) + " " + str(neg_num))
            ################################# find controversial balance of movies ####################
            # if neg_num == 0: print(" balance: ")
            # else:
            #     balance = pos_num/1.0/neg_num
            #     if balance < 1 and balance > 0: balance = 1/balance
            #     sum_balance += balance
            #     num_balance += 1
            #     print(" balance: "+str(balance) + " num: "+str(len(review_person)))
            # pos_num = 0
            # neg_num = 0


            ######This should always happen in the end######
            num_aspect = 0
            review_list.append(all_review) #each element is all the reviews under a movie
            all_review = review['reviewText']  # to start new one
            prev_product = product_id
            product_num += 1    #product number plus one"
            #review_each_movie.append(review_person)
            review_person = []
            review_person.append(review['reviewText'])
            ################analyze each review which aspects are inluded#####################
            # score_array = np.array([similarity.aspect_included(num_review, prev_product, review['reviewText'], 0.471, "minmax")])  # 0.481 for avg
            ################analyze each review which aspects are inluded#####################
        # else:
        #     pass
    
        if product_num == 2000:    #after the product_num, break the for loop, always -1
            ############try to use alchemyapi#######################
            # import new_test as alchemy_test
            # alchemy_test.start_queryAlchemy(" ".join(review_list).lower())
            #####################topic_modelling(review_list)#####################
#             ##### adjective+noun/ noun+adjective ####
#             count_words = {}
#             for word in set(possible_words):
#                 if word not in ["movie", "film", "s", "t"]:
#                     count_words[word] = possible_words.count(word)
#             sorted_words = sorted(count_words.items(), key=operator.itemgetter(1)) 
#             for each in sorted_words:
#                 print each
            #####################select aspect words#####################
            aggregated_review = " ".join(review_list).lower().translate(str.maketrans('','',string.punctuation))
            review_list = []
            #print (aggregated_review)
            tf_idf(aggregated_review)


            #####################try clustering#####################

#             tag_aggregated_review = nltk.pos_tag(tokenizer.tokenize(this_review))
#             noun_aggregated_review = [word for (word,tag) in tag_aggregated_review if "NN" in tag]
            #print aggregated_review
            

#             for word in set(tokenizer.tokenize(" ".join(review_list).lower())):
#                 corpus.append(lmtzr.lemmatize(word))
#
#             noun_corpus = set([word for (word, tag) in nltk.pos_tag(corpus) if "NN" in tag])
#             print noun_corpus
#             for word1 in noun_corpus:
#                 for word2 in noun_corpus:
#                     similarity = para.paradigSimilarity(word1, word2)    #association rules to find similar words
#                     if similarity > 0 and similarity < 1: print word1, word2, similarity

            break

    #print (product_num)

def tf_idf(aggregated_review):
    import TFIDF as tfidf
    print ("##########Start TF-IDF......##############")
    tfidf_dict = tfidf.tfidf_result(aggregated_review)
    words_tfidf = [word for word in tfidf_dict if word not in ["ha", "wa",  "im", "ive", "don't", "doesn't", "dont", "doesnt"]]
    tag_tfidf = nltk.pos_tag(words_tfidf)
    noun_tfidf = [word for (word,tag) in tag_tfidf if "NN" in tag]
    print (noun_tfidf)
    noun_tfidf_score = {}
    for word in noun_tfidf:
        score = tfidf_dict[word]
        noun_tfidf_score[word] = score

                
    sorted_tfidf = sorted(noun_tfidf_score.items(), key=operator.itemgetter(1))
    for word in sorted_tfidf:
        print (word)
        
def sentiment_analyze(review_tag, sentence_split):
    aspect_list = ["acting", "directing", "scene", "character", "story"]
    #aspect_list = ["acting"]
    scores = []
    for aspect in aspect_list:
        print ("************"+ aspect + "************")
        print ("Total Score: " , score)
        scores.append(score)

    return scores


def visualWords():
    import networkx as nx
    import matplotlib.pyplot as pl
    from nltk.corpus import wordnet as wn
    dog = wn.synset('dog.n.01')
    graph = closure_graph(dog,
                          lambda s: s.hypernyms())
    nx.draw_graphviz(graph)

#visualWords()
#start_analyze()
aspect_rank()





