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

# Getting all codeup blog urls in one function:
def get_all_urls():
    
    url = 'https://codeup.com/resources/#blog'
    
    soup = make_soup(url)
    
    urls_list = soup.find_all('a', class_='jet-listing-dynamic-link__link')
    
    urls = {link.get('href') for link in urls_list}

    urls = list(urls)
        
    return urls


# This is the full function which looks through all of the codeup blog to retrieve the articles posted:

def get_blog_articles(urls, cached=False):
    '''
    This function takes in a list of Codeup Blog urls and a parameter
    with default cached == False which scrapes the title and text for each url, 
    creates a list of dictionaries with the title and text for each blog, 
    converts list to df, and returns df.
    If cached == True, the function returns a df from a json file.
    '''
    if cached == True:
        df = pd.read_json('big_blogs.json')
        
    # cached == False completes a fresh scrape for df     
    else:

        # Create an empty list to hold dictionaries
        articles = []

        # Loop through each url in our list of urls
        for url in urls:

            # Make request and soup object using helper
            soup = make_soup(url)

            # Save the title of each blog in variable title
            title = soup.find('h1').text

            # Save the text in each blog to variable text
            content = soup.find('div', class_="jupiterx-post-content").text

            # Create a dictionary holding the title and content for each blog
            article = {'title': title, 'content': content}

            # Add each dictionary to the articles list of dictionaries
            articles.append(article)
            
        # convert our list of dictionaries to a df
        df = pd.DataFrame(articles)

        # Write df to a json file for faster access
        df.to_json('big_blogs.json')
    
    return df



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

# Create a helper function that requests and parses HTML returning a soup object.

def make_soup(url):
    '''
    This helper function takes in a url and requests and parses HTML
    returning a soup object.
    '''
    headers = {'User-Agent': 'Codeup Data Science'} 
    response = get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    return soup


# Solved function:

def get_news_articles(cached=False):
    '''
    This function with default cached == False does a fresh scrape of inshort pages with topics 
    business, sports, technology, and entertainment and writes the returned df to a json file.
    cached == True returns a df read in from a json file.
    '''
    # option to read in a json file instead of scrape for df
    if cached == True:
        df = pd.read_json('articles.json')
        
    # cached == False completes a fresh scrape for df    
    else:
    
        # Set base_url that will be used in get request
        base_url = 'https://inshorts.com/en/read/'
        
        # List of topics to scrape
        topics = ['business', 'sports', 'technology', 'entertainment']
        
        # Create an empty list, articles, to hold our dictionaries
        articles = []

        for topic in topics:
            
            # Create url with topic endpoint
            topic_url = base_url + topic
            
            # Make request and soup object using helper
            soup = make_soup(topic_url)

            # Scrape a ResultSet of all the news cards on the page
            cards = soup.find_all('div', class_='news-card')

            # Loop through each news card on the page and get what we want
            for card in cards:
                title = card.find('span', itemprop='headline' ).text
                author = card.find('span', class_='author').text
                content = card.find('div', itemprop='articleBody').text

                # Create a dictionary, article, for each news card
                article = ({'topic': topic, 
                            'title': title, 
                            'author': author, 
                            'content': content})

                # Add the dictionary, article, to our list of dictionaries, articles.
                articles.append(article)
            
        # Create a DataFrame from list of dictionaries
        df = pd.DataFrame(articles)
        
        # Write df to json file for future use
        df.to_json('articles.json')
    
    return df

print("All acquire functions loaded properly.")