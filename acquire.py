# Importing Libraries
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

# 1st Acquire function:

def get_article_text(url):
    # if we already have the data, read it locally
    if os.path.exists('article.txt'):
        with open('article.txt') as f:
            return f.read()

    # otherwise go fetch the data
    url = url
    headers = {'User-Agent': 'Codeup Data Science'}
    response = get(url, headers=headers)
    soup = BeautifulSoup(response.text)
    article = soup.find('div', class_='jupiterx-post-content')

    # save it for next time
    with open('article.txt', 'w') as f:
        f.write(article.text)

    return article.text



# First raw attempt at inshorts news scraping:

urls = ['https://codeup.com/codeups-data-science-career-accelerator-is-here/', 'https://codeup.com/data-science-myths/', 'https://codeup.com/data-science-vs-data-analytics-whats-the-difference/', 'https://codeup.com/10-tips-to-crush-it-at-the-sa-tech-job-fair/','https://codeup.com/competitor-bootcamps-are-closing-is-the-model-in-danger/']
def get_blog_articles(url_list):
    final = [] 
    for x in url_list:
        url = x
        headers = {'User-Agent': 'Codeup Data Science'} # Some websites don't accept the pyhon-requests default user-agent
        response = get(url, headers=headers)
        soup = BeautifulSoup(response.content, 'html.parser')
        article_title = soup.title.string
        article = soup.find('div', class_='jupiterx-post-content')
        article_text = article.text
        item = {
            'title': article_title,
            'content': article_text
        }
        final.append(item)
            # save it for next time
    with open('article.txt', 'w') as f:
        f.write(article.text)
    return final

# Big thanks to Matt for his help getting me pointed in the right direction on this!


# Using a function that allows one to feed the source URLs

source_urls = ['https://inshorts.com/en/read/technology',
             'https://inshorts.com/en/read/sports',
             'https://inshorts.com/en/read/business',
             'https://inshorts.com/en/read/entertainment']

def build_dataset(source_urls):
    news_data = []
    for url in source_urls:
        news_category = url.split('/')[-1]
        data = get(url)
        soup = BeautifulSoup(data.content, 'html.parser')
        
        news_articles = [{'title': headline.find('span', 
                                                         attrs={"itemprop": "headline"}).string,
                          'content': article.find('div', 
                                                       attrs={"itemprop": "articleBody"}).string,
                          'category': news_category}
                         
                            for headline, article in 
                             zip(soup.find_all('div', 
                                               class_=["news-card-title news-right-box"]),
                                 soup.find_all('div', 
                                               class_=["news-card-content news-right-box"]))
                        ]
        news_data.extend(news_articles)
        
    df =  pd.DataFrame(news_data)
    df = df[['title', 'content', 'category']]
    return df

print("All acquire functions loaded properly.")