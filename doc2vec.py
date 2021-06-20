from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from nltk.tokenize import word_tokenize
import pandas as pd
from nltk.corpus import stopwords
import random
stopWordsEnglish = stopwords.words('english')

df = pd.read_csv('./spam_or_not_spam/spam_or_not_spam.csv', encoding='utf8')

# get all emails
allEmail = df.email
allLabel = df.label
sentences = ['' for x in range(len(allEmail))]
training = []

# Convert emails to strings
df['stringEmail'] = df['email'].astype(str)
# Replace unencoded values
#df['stringEmail'].replace(regex=r'\\', value='')
# Tokenize email 
df['token'] = df['stringEmail'].apply(word_tokenize)
# Remove stopwords
df['without_stopwords'] = df['token'].apply(lambda x: [item.lower() for item in x if item.lower() not in stopWordsEnglish])

sentences = df['without_stopwords'] # separated tokens

shuffleTemp = list(zip(sentences, allLabel))
random.shuffle(shuffleTemp)
sentences , allLabel = zip(*shuffleTemp)

model = Doc2Vec(min_count=1, window=10, vector_size=100, sample=1e-4, negative=5, workers=7)

for i in range(len(sentences)):
    training.append(TaggedDocument(words=sentences[i], tags=[i]))

model.build_vocab(training)

for epoch in range(10):
    print(epoch)
    model.train(training,
                total_examples=model.corpus_count,
                epochs=model.epochs)
model.save('./doc2vecModel.d2v')
