from gensim.models import Doc2Vec
import nltk
from nltk.corpus import brown
from gensim.models import Word2Vec
import multiprocessing
import sys
from gensim.models.doc2vec import TaggedDocument
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
import pandas as pd
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from numpy import loadtxt
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential
from tensorflow.keras import backend as K

model = Doc2Vec.load('./spamModel.d2v')

stopWordsEnglish = stopwords.words('english')


def splitTrainTest(percentage, X, Y):
    lengthOfTrain = int((percentage*len(X))/100)
    X_train = X[:lengthOfTrain]
    Y_train = Y[:lengthOfTrain]
    X_test = X[lengthOfTrain:]
    Y_test = Y[lengthOfTrain:]
    return X_train, X_test, Y_train, Y_test

df = pd.read_csv('./spam_or_not_spam/spam_or_not_spam.csv', encoding='utf8')

# get all emails
allEmail = df.email
allLabel = df.label
sentences = ['' for x in range(len(allEmail))]
vectorizedEmail = ['' for x in range(len(allEmail))]
training = []

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

allEmailClean = df['without_stopwords'].apply(lambda x: ' '.join(x)) # concat sentences 
sentences = df['without_stopwords'] # separated tokens


for i in range(4):
    inferred_embedding = model.infer_vector(sentences[i])
    print(len(inferred_embedding))

model = Sequential()
model.add(Dense(12, input_dim=100, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

[X_train, X_test, Y_train, Y_test] = splitTrainTest(75, allEmail, allLabel)

print(Y_train.shape)
