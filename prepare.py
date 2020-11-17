from requests import get
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

import unicodedata
import re
import json

import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords

import acquire

nltk.download('wordnet')

source_urls = ['https://inshorts.com/en/read/technology',
             'https://inshorts.com/en/read/sports',
             'https://inshorts.com/en/read/business',
             'https://inshorts.com/en/read/entertainment']


# Basic text cleaner:
def basic_clean(text):
    
    text = text.lower()
    recode = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    clean_text = re.sub(r"[^a-z0-9'\s]", '', recode)
    clean_text_final = re.sub(r"\n", "", clean_text)
    
    return clean_text_final


# Tokenize:

def tokenize(text):
    
    tokenizer = nltk.tokenize.ToktokTokenizer()

    text = tokenizer.tokenize(text, return_str=True)
    
    return text


# Stemming the text:


def stem(text):
    
    ps = nltk.porter.PorterStemmer()
    stems = [ps.stem(word) for word in text.split()]
    text_stemmed = ' '.join(stems)
    
    return text_stemmed


def lemmatize(string):
    nltk.download('wordnet')


    # Defining the lemmatizer:
    wnl = nltk.stem.WordNetLemmatizer()
    
    lemmas_text = [wnl.lemmatize(word) for word in string.split()]
    
    string_final = ' '.join(lemmas_text)
    return string_final


# Removing stopwords and adding more words to exclude list:

def remove_stopwords(string, extra_words = [], exclude_words = []):
    
    # Stopwords list:
    stopword_list = stopwords.words('english')
    
    # Remove any excluded words:
    stopword_list = set(stopword_list) - set(exclude_words)
    
    stopword_list = stopword_list.union(set(extra_words))
    
    words = string.split()
    
    filtered_words = [word for word in words if word not in stopword_list]
    
    no_stopwords = ' '.join(filtered_words)
    
    return no_stopwords


# Full prep function (per Faith's walkthrough):

def prep_article_data(df, column, extra_words=[], exclude_words=[]):

    df['clean'] = df[column].apply(basic_clean)\
                            .apply(tokenize)\
                            .apply(remove_stopwords, 
                                   extra_words=extra_words, 
                                   exclude_words=exclude_words)\
                            .apply(lemmatize)
    
    df['stemmed'] = df[column].apply(basic_clean).apply(stem)
    
    df['lemmatized'] = df[column].apply(basic_clean).apply(lemmatize)
    
    return df[['title', column, 'stemmed', 'lemmatized', 'clean']]



print("All prepare functions successfully loaded.")