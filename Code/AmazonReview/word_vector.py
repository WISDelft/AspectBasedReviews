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

def start_word_vector():
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
    review_each_movie = []
    tokenizer = RegexpTokenizer(r'\w+')
    stop_words = set(stopwords.words('english'))
    nouns_verbs_dic = {}
    corpus = []  #all the words appeared in reviews, no duplicates
    possible_words = []

    # with open('score.txt', 'w') as f:
    #             f.write("id, id2, acting, directing, scene, character, story, \n")

    for i, review in enumerate(parse("reviews_Movies_and_TV_5.json.gz")):
        #Select one movie only
        #print review['reviewText']
        product_id = review['asin']
        #Choose a specific product number, first 0005019281, second product 0005119367, 
        this_review = review['reviewText']

        if (product_id != prev_product):
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
            print("*" * 30, product_num, "th product", product_id, "*" * 30)
            num_review = 0
            # with open(filename, 'a') as f:
            #     f.write("*" * 30+str(product_num) + "th product" +product_id +"*" * 30)
            #     f.write("\n")

        if (product_id == prev_product): num_review += 1
        #print(str(num_review) + " " + this_review)


        ############limitation, restrict################
        # if len(review_person) >= 20 and prev_product == product_id:      ### to make sure 10 reviews for each movie
        #     if (len(review_person)) > 20:
        #         continue;

            ################analyze each review which aspects are inluded#####################
        # if (prev_product) == '':        #initialize the value
        #     score_array = np.array([similarity.aspect_included(num_review, product_id, this_review, 0.471, "minmax")])  # 0.481 for avg
        #
        #
        #
        # if (product_id == prev_product):
        #     aspect_score = np.array([similarity.aspect_included(num_review, product_id, this_review, 0.471, "minmax")])   # 0.481 for avg
        #     score_array = np.concatenate((score_array, np.array(aspect_score)))
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
            #print prev_product, product_id
        elif product_id != prev_product: #product_id no equals to previous product
            #print prev_product, product_id
            #print prev_product, product_id
            #print all_review
            #review_person.append(review['reviewText'])



            #####to analyze the sentiment of the aspect######
            # translator = str.maketrans({key: None for key in string.punctuation})
            # split_review = all_review.lower().translate(translator).split()
            # tag_each_movie = nltk.pos_tag(split_review)
            # sentence_split = all_review.lower().split(".")
            # (scores, num) = sentiment_analyze(tag_each_movie, sentence_split)
            # write_score = prev_product + ", " + str(num) + ", " + str(num_review) + ", "
            # for i in range(0, len(scores)):
            #     if i < len(scores)-1:
            #         write_score += str(scores[i])+", "
            #     else:
            #         write_score += str(scores[i])
            # with open('score.txt', 'a') as f:
            #     f.write(write_score + "\n")



            ######This should always happen in the end######
            review_list.append(all_review) #each element is all the reviews under a movie
            all_review = review['reviewText']  # to start new one
            prev_product = product_id
            product_num += 1    #product number plus one"
            #review_each_movie.append(review_person)
            #review_person = []
            review_person.append(review['reviewText'])
            ################analyze each review which aspects are inluded#####################
            #score_array = np.array([similarity.aspect_included(num_review, prev_product, review['reviewText'], 0.471, "minmax")])  # 0.481 for avg
            ################analyze each review which aspects are inluded#####################
        # else:
        #     pass
    
        if product_num == 1000:    #after the product_num, break the for loop, always -1

            #####################try gensim word2vec########################
            import gensim, logging
            logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
            split_word_persons = []
            for each_review in review_person:
                characters = [letter.lower() if letter.isalpha() else ' ' for letter in each_review]
                sentence = "".join(characters)
                split_word_persons.append([lmtzr.lemmatize(word) for word in sentence.split()])
            # train word2vec on the two sentences
            print(split_word_persons)
            model = gensim.models.Word2Vec(split_word_persons, min_count=4)

            #print(model.similarity('classic', 'happen'))
            return model
            #####################try gensim word2vec########################




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
            # aggregated_review = " ".join(review_list).lower().translate(str.maketrans('','',string.punctuation))
            # print (aggregated_review)
            # tf_idf(aggregated_review)


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




