import nltk
import string
import os
from nltk.corpus import brown
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
from sklearn.datasets import fetch_20newsgroups
from nltk.stem.wordnet import WordNetLemmatizer

def tokenize(text):
    lmtzr = WordNetLemmatizer()
    tokens = nltk.word_tokenize(text)
    not_wanted_words =  ["movie", "film", "movies", "films"]
    stems = [lmtzr.lemmatize(word) for word in tokens if word not in not_wanted_words]
    return stems

def tfidf_result(raw_text):
    newsgroups_train = fetch_20newsgroups(subset='train')
    vectorizer = TfidfVectorizer(tokenizer=tokenize, stop_words='english')
    vectors = vectorizer.fit_transform(newsgroups_train.data)
    response = vectorizer.transform([raw_text])
    feature_names = vectorizer.get_feature_names()
    tfidf_dict = {}
    for col in response.nonzero()[1]:
        tfidf_dict[feature_names[col]] = response[0, col]
    
    return tfidf_dict
  