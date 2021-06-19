import nltk
from nltk.corpus import brown
from gensim.models import Word2Vec
import multiprocessing
import sys
from nltk.tokenize import word_tokenize
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec

df = pd.read_csv('./spam_or_not_spam/spam_or_not_spam.csv', encoding='utf8')
df.head()
# get all emails
allEmail = df.email
sentences = ['' for x in range(len(allEmail))]
vectorizedEmail = ['' for x in range(len(allEmail))]
i=0
for i in range(len(allEmail)):
    sentences[i] = word_tokenize(str(allEmail[i]))
    #print(len(sentences[i]))
"""
# Vectorizer
vectorizer = TfidfVectorizer()

# sentences array has every tokenized email
corpus = ["hello this is paulinho"]
vectorizer.fit(corpus)
temp = vectorizer.transform(corpus)
print(temp)"""
print(sentences[0])
model = Word2Vec(sentences=sentences, min_count=1, vector_size= 100, workers=3, window =5, sg = 1)
print(model.wv['said'])
print(model.wv.most_similar('said', topn=10) )
