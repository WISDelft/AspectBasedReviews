from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import gensim
from gensim import corpora, models, similarities
from gensim.models.ldamodel import LdaModel
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus.reader.wordnet import WordNetError
import numpy as np

lmtzr = WordNetLemmatizer()
snow_stem = SnowballStemmer("english")
tokenizer = RegexpTokenizer(r'\w+')
act_aspect = ['act','acting','actress','actor','portray', 'role', 'character', 'villain', 'performance', 
              'performed', 'played', 'casting','cast']
act_stem = [lmtzr.lemmatize(word) for word in act_aspect] #stem the act aspect words
stop_words = set(stopwords.words('english'))

def get_word_similarity(word1, word2, min_max):
    noun1 = lmtzr.lemmatize(word1, "n")
    verb1 = lmtzr.lemmatize(word1,"v")
    noun2 = (lmtzr.lemmatize(word2,"n"))
    verb2 = lmtzr.lemmatize(word2,"v")

    i = 1
    type = "n"
    type_word = noun1
    sets1 = []

    #for example: acting and act, should be the same meaning
    if (word1 == word2):
        return 1.0

    while (True):
        try:
            set = wn.synset(type_word+"."+type+"." + str(i))
            sets1.append(set)
        except (WordNetError):
            # if type != "v":
            #     type = "v"
            #     type_word = verb1
            #     i = 0
            #     continue
            # else:
            #     break
            break;
        i += 1

    i = 1
    sets2 = []
    type = "n"
    type_word = noun2
    while (True):
        try:
            set = wn.synset(type_word+"."+type+"." + str(i))
            sets2.append(set)
        except (WordNetError):
            # if type != "v":
            #     type = "v"
            #     type_word = verb2
            #     i = 0
            #     continue
            # else:
            #     break
            break;
        i += 1

    avg = 0.0
    num = 0.0
    min = 2
    max = 0
    second_max = 0
    for set1 in sets1:
        for set2 in sets2:
            #print (set1, set2)
            similarity = set1.wup_similarity(set2)
            #print (similarity)
            if similarity != None:
                avg += similarity
                num += 1
            else: similarity = 0
            if similarity > max:
                second_max = max
                max = similarity
            if similarity < min:
                min = similarity


    if num == 0:
        return 0.0
    if min_max == "avg":
        return avg/num
    if min_max == "2max":
        return (max+second_max)/2.0
    if min_max == "min":
        return min
    if min_max == "max":
        #print (max)
        return max
    if min_max == "minmax":
        return (min+max)/2.0

    
def get_similarity(sets1, sets2, min_max):
    max_similarity = 0;
    min_similarity = 1;
    avg = 0.0;
    num = 0.0;
    sets1 = sets1[:2]
    sets2 = sets2[:2]
    if min_max == "one" and len(sets1) >=1 and len(sets2) >= 0:
        return sets1[0].wup_similarity(sets2[0])
    elif min_max == "one":
        return 0
    for set1 in sets1:
        for set2 in sets2:            
            similarity = set1.wup_similarity(set2)  #lch_similarity
            if similarity != None:
                avg += similarity
                num += 1
            else: similarity = 0
            if similarity > max_similarity:
                max_similarity = similarity
            if similarity < min_similarity:
                min_similarity = similarity    
    if num == 0:
        return 0
    else:
        avg = avg / num
    if min_max == "min":
        return min_similarity
    elif min_max == "max":
        return max_similarity
    elif min_max == "avg":
        return avg



def manual_count_score(aspects_stem, review_with_tags, window_length):
    half_window = window_length/2
    count_act = 0
    total = 0
    for i, (word,tag) in enumerate(review_with_tags):
        if word.lower() in act_stem:
            window = review_with_tags[i-half_window:i+half_window+1]    #window is 5 gram
            
            score = 0;
            for j in range(0, len(window)):
                if j < len(window):
                    synset = []
                    if ('JJ' in window[j][1]): #JJ is adjective, RB is adveb
                        synset = list(swn.senti_synsets(window[j][0], 'a'))
                    if ("RB" in window[j][1]):
                        synset = list(swn.senti_synsets(window[j][0], 'r'))
                    if len(synset) != 0:
                        score += sum(syn.pos_score()-syn.neg_score() for syn in synset)
                        #score will be reversed if not of n't detected
#                         if "not" in window or "n't" in window:
#                             score = -score
                        if score < 0:
                          score = 0
                        total += score
                        count_act += 1
             
            #print window, score
                  
                  
    return total          
#     print count_act
#     print total

def aspect_included(num_review, id, review, thresh, min_max):
    import string
    import nltk

    translator = str.maketrans({key: None for key in string.punctuation})
    sentence_split = review.lower().replace(",", ".").replace("!", ".").replace("?", ".").replace(";", ".").split(".")
    #print (sentence_split)
    split_review = review.lower().translate(translator).split()
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words('english'))
    stop_words.add("movie")
    stop_words.add("film")
    stop_words.add("movies")
    stop_words.add("films")
    aspect_list = ["acting", "directing", "scene", "character", "story"]
    appeared_aspects = []
    aspect_score = []
    #print (review)
    aspect_synsets = []
    adj_words = []
    for aspect in aspect_list:
        score = 0.0
        max = 0.0
        num = 0.0
        words = []
        for sentence in sentence_split:
            for i, (word,tag) in enumerate(nltk.pos_tag(sentence.translate(translator).split())):
                if word not in stop_words not in stop_words and len(word) != 1:
                    similarity = get_word_similarity(aspect, word, min_max)
                    if similarity > thresh:
                        if similarity == 1 or "NN" in tag:
                            #print(aspect, word, similarity)
                            (temp_score, temp_num, words) = sentiment_score(sentence, 7,2, similarity, i)
                            if (temp_score != None):
                                score += temp_score
                                if temp_score > max: max = temp_score
                                num += temp_num
                                break;  # avoid aspect word happen again in the same sentence

        words = set(words)
        adj_words.append(" ".join(words))
        if num != 0:
            aspect_score.append(max)
            #aspect_score.append(score/num)
        else:
            import numpy as np
            aspect_score.append(None)   #missing value
            #aspect_score.append(2.5)    #the aspect is not mentioned means the writer is neautral


    result = ""
    toWrite = id+"," + str(num_review) +","
    for i, aspect in enumerate(aspect_list):
        result += aspect +"; " + str(aspect_score[i]) +  " "
        toWrite += str(aspect_score[i]) + ","
        #if i < len(aspect_list)-1: toWrite += ","

    #print (adj_words)
    toWrite += (",".join(adj_words))
    toWrite += "\n"
    #print (result)
    with open('score.csv', 'a') as f:
        f.write(toWrite)
    #print(toWrite)

    #print ()
    return (aspect_score)

def sentiment_score(sentence, left_length, right_length, similarity, word_location):
    import string
    import nltk
    score = 0.0
    num = 0.0
    max = 0.0
    translator = str.maketrans({key: None for key in string.punctuation})
    sentence_word = sentence.translate(translator).split()
    tag_word = nltk.pos_tag(sentence_word)
    left = int(word_location-left_length)
    right = int(word_location+right_length+1)
    if (left < 0):
        left = 0
    if (right > len(tag_word)):
        right = len(tag_word)

    window = tag_word[left : right]
    has_words = False
    words = []

    for j, (word, tag) in enumerate(window):
        if word in stop_words:
            continue
        synset = []
        synset_JJ = []
        synset_RB = []
        #
        # if ('JJ' in tag):  # JJ is adjective, RB is adveb, "a" "r" in wordnet respectively
        #     synset_JJ = list(swn.senti_synsets(word, 'a'))
        #     print (word)
        # if ("RB" in tag):
        #     synset_RB = list(swn.senti_synsets(word, 'r'))
        #     print(word)
        # synset = synset_JJ + synset_RB

        type = ['a', 'r']
        #type = ['a', 'r', 'v', 's']
        i = 0
        number = 1
        while (True):
            try:
                set = swn.senti_synset(word + "." + type[i] + "." + str(number))
                senti_score = set.pos_score() - set.neg_score();
                # if abs(set.pos_score() - set.neg_score()) > 0.2:    #ignore objective words:
                synset.append(set)
                if abs(senti_score) > 0.25: words.append(word)
                #print(type[i], number, set)
                number += 1
            except (WordNetError):
                i += 1
                number = 1
                if i >= len(type):
                    break


        num_local = 0.0
        #print (str(len(synset)))
        if len(synset) != 0:
            for syn in synset:
                compare_score = set.pos_score() - set.neg_score() ;
                if abs(compare_score) > max:
                    max = compare_score
                    has_words = True
                    num +=1
                #if compare_score > 0.4: words.append(word)
            word_score = sum(1 + syn.pos_score() - syn.neg_score()  for syn in synset )/len(synset) #average sentiment of this word
            #print(word, ' ', word_score, word_score * 2.5)
            #if abs(word_score-1) >= 0.25:
            # if abs(word_score) >= 0.25:
            #     has_words = True
            #     num += 1  # for all words counting
            #     score += (word_score * 2.5)
            #     words.append(word)

            # score will be reversed if not of n't detected
    if not has_words:
        return (None, None, [])
    #print (str(words) + " " +str(score))
    return ((max+1) * 2.5, num, words)
    #return (score, num, words)


#return the counted sentiment score based on the similarity of words 
def count_score(aspect, review_with_tag, left, right,sentence_split, thresh, min_max):
    """
    Return the score of a aspect in a review
    @window_length should be int, the length of the window to find adjective
    @thresh the threshhold, how similar should the words be
    """
    aspect_sets = wn.synsets(aspect)
    num = 0.0
    score = 0.0
    aspect = lmtzr.lemmatize(aspect)
    do_not_want = ["movie", "film"]
    import string
    import nltk
    translator = str.maketrans({key: None for key in string.punctuation})
    for i, (word, tag) in enumerate(review_with_tag):
        
        similarity = 0
        word = lmtzr.lemmatize(word)
        
        if word not in do_not_want and word not in stop_words:
            word_sets = wn.synsets(word)
            #similarity = get_similarity(aspect_sets, word_sets, "max")
            similarity = get_word_similarity(aspect, word, min_max)

        for sentence in sentence_split:
            for i, (word, tag) in enumerate(nltk.pos_tag(sentence.translate(translator).split())):
                if "NN" in tag and word not in stop_words and len(word) != 1:
                    temp_score = sentiment_score(sentence, left, right, similarity, i)
                    if temp_score != None:
                        score += temp_score
                        num += 1
                        break

             
            #print window, score
                           
              
    #print count_act
    #print total
    
    if num != 0:
        return score/num
    else:
        return None


def topic_modelling(reviews,num_topics, num_words):
    # Use topic modelling to generate topic
        texts = []
        for each_review in reviews:
            tokens = tokenizer.tokenize(each_review.lower())
            non_stop_tokens = [x for x in tokens if x not in stop_words and x not in ['movie', 'film', 'movies', 'films', 'dvd', 'blue', 'ray']] 
            stemmed_tokens = [lmtzr.lemmatize(x) for x in non_stop_tokens]
            texts.append(stemmed_tokens)

        dictionary = corpora.Dictionary(texts)
        corpus = [dictionary.doc2bow(text) for text in texts]
        #model = LsiModel(corpus, id2word=dictionary, num_topics=300)
        #model = models.LsiModel(tfidf_corpus, id2word=dictionary, num_topics=300)]
        ldamodel = LdaModel(corpus, num_topics, id2word = dictionary, passes=20)
        print (ldamodel.print_topics(num_topics, num_words))


#print (get_word_similarity("acting", "acting","minmax"))