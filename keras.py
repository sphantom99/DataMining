from gensim.models import Doc2Vec
import numpy
from nltk.tokenize import word_tokenize
import pandas as pd
from nltk.corpus import stopwords
from tensorflow.keras.layers import Dense
from tensorflow.keras import Sequential
from tensorflow.keras import backend as K
import random
model = Doc2Vec.load('./doc2vecModel.d2v')
stopWordsEnglish = stopwords.words('english')

df = pd.read_csv('./spam_or_not_spam/spam_or_not_spam.csv', encoding='utf8')

# get all emails
allEmail = df.email
allLabel = df.label
sentences = ['' for x in range(len(allEmail))]

def splitTrainTest(percentage, X, Y):
    lengthOfTrain = int((percentage*len(X))/100)
    X_train = X[:lengthOfTrain]
    Y_train = Y[:lengthOfTrain]
    X_test = X[lengthOfTrain:]
    Y_test = Y[lengthOfTrain:]
    return X_train, X_test, Y_train, Y_test
    
# Convert emails to strings
df['stringEmail'] = df['email'].astype(str)
# Tokenize email 
df['token'] = df['stringEmail'].apply(word_tokenize)
# Remove stopwords
df['without_stopwords'] = df['token'].apply(lambda x: [item.lower() for item in x if item.lower() not in stopWordsEnglish])

sentences = df['without_stopwords'] # separated tokens
# Shuffle dataset so it is more random,
# Both columns are shuffled
shuffleTemp = list(zip(sentences, allLabel))
random.shuffle(shuffleTemp)
sentences , allLabel = zip(*shuffleTemp)

# Imported doc2vec model, and converted tokenized email into vectors
inferred_embedding =[]
for i in range(len(sentences)):
    inferred_embedding.append(model.infer_vector(sentences[i]))

def splitTrainTest(percentage, X, Y):
    lengthOfTrain = int((percentage*len(X))/100)
    X_train = X[:lengthOfTrain]
    Y_train = Y[:lengthOfTrain]
    X_test = X[lengthOfTrain:]
    Y_test = Y[lengthOfTrain:]
    return X_train, X_test, Y_train, Y_test

def recall_metric(y_true, y_pred):
    true_positive = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positive / (possible_positives + K.epsilon())
    return recall

def precision_metric(y_true, y_pred):
    true_positive = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positive / (predicted_positives + K.epsilon())
    return precision

def f1_metric(y_true, y_pred):
    precision = precision_metric(y_true, y_pred)
    recall = recall_metric(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))

model = Sequential()
model.add(Dense(14, input_dim=100, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', 
              optimizer='adam', 
              metrics=['accuracy', 
                       f1_metric, 
                       precision_metric, 
                       recall_metric])

[X_train, X_test, Y_train, Y_test] = splitTrainTest(75, inferred_embedding, allLabel)

model.fit(numpy.array(X_train), 
          numpy.array(Y_train), 
          epochs=10, 
          batch_size=20,
          validation_data=(numpy.array(X_test), numpy.array(Y_test)))

model.save('./kerasModel')
