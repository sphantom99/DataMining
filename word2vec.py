import nltk
from nltk.corpus import brown
from gensim.models import Word2Vec
import multiprocessing
import sys
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
stopWordsEnglish = stopwords.words('english')


df = pd.read_csv('./spam_or_not_spam/spam_or_not_spam.csv', encoding='utf8')

# get all emails
allEmail = df.email
allLabel = df.label
sentences = ['' for x in range(len(allEmail))]
vectorizedEmail = ['' for x in range(len(allEmail))]

def splitTrainTest(percentage, X, Y):
    lengthOfTrain = int((percentage*len(X))/100)
    X_train = X[:lengthOfTrain]
    Y_train = Y[:lengthOfTrain]
    X_test = X[lengthOfTrain:]
    Y_test = Y[lengthOfTrain:]
    return X_train, X_test, Y_train, Y_test




# Convert emails to strings
df['stringEmail'] = df['email'].astype(str)
# Replace unencoded values
df['stringEmail'].replace(regex=r'\\', value='')
# Tokenize email 
df['token'] = df['stringEmail'].apply(word_tokenize)
# Remove stopwords
df['without_stopwords'] = df['token'].apply(lambda x: [item.lower() for item in x if item.lower() not in stopWordsEnglish])
# Join email tokens

allEmailClean = df['without_stopwords'].apply(lambda x: ' '.join(x))
sentences = df['without_stopwords']



[X_train, X_test, Y_train, Y_test] = splitTrainTest(75, allEmail, allLabel)

model = Word2Vec(sentences=sentences, min_count=1, vector_size= 3, workers=3, window =8, sg = 1)
voc = model.wv
for i in range(len(sentences)):
    for y in range(len(sentences[i])):
        sentences[i][y]=model.wv[sentences[i][y]]
        
print(sentences[0])

print(model.wv['said'])
print(model.wv['hello'])
#print(model.wv.most_similar('said', topn=10) )
print(voc.similarity('said', 'hello'))





































"""
print(sent_tokenize(allEmail[0]))


# Tokenize all emails
for i in range(len(allEmail)):
    sentences[i] = word_tokenize(str(allEmail[i]))
    #print(len(sentences[i]))


# Vectorizer
vectorizer = TfidfVectorizer()

# sentences array has every tokenized email
corpus = ["hello this is paulinho"]
vectorizer.fit(corpus)
temp = vectorizer.transform(corpus)
print(temp)
# Vectorize all emails"""


