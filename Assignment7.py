#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag

# Sample document
document = '''
Natural language processing (NLP) is a subfield of artificial intelligence and computational linguistics 
that focuses on the interactions between computers and human language. NLP techniques are used to analyze, 
understand, and derive meaning from text and speech data. Tokenization is the process of splitting text into 
individual words or tokens. POS tagging assigns a grammatical label (part-of-speech) to each token. Stop words 
are common words that are often removed from text as they do not carry significant meaning. Stemming reduces 
words to their base or root form. Lemmatization is a more advanced technique that reduces words to their 
base form (lemma) based on their dictionary meaning.'''

# Tokenization
tokens = word_tokenize(document)
print("Tokens:", tokens)

# POS tagging
pos_tags = pos_tag(tokens)
print("POS Tags:", pos_tags)

# Stop words removal
stop_words = set(stopwords.words('english'))
filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
print("Tokens after Stop words removal:", filtered_tokens)

# Stemming
stemmer = PorterStemmer()
stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
print("Stemmed Tokens:", stemmed_tokens)

# Lemmatization
lemmatizer = WordNetLemmatizer()
lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
print("Lemmatized Tokens:", lemmatized_tokens)

