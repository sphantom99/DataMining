from tensorflow import keras
from gensim.models import Doc2Vec
import nltk
import numpy
from nltk.tokenize import word_tokenize
modelKeras = keras.models.load_model('./kerasModel')
modelD2V = Doc2Vec.load('./spamModel.d2v')

def cleanInput(toPredict):
    temp = str(toPredict)
    temp = temp.lower()
    temp = word_tokenize(temp)
    temp = modelD2V.infer_vector(temp)
    return temp

#val = input("Enter sentence to predict: ")
#print(val)
val = "hello my name is paulinho"
input = cleanInput(val)
print(len(input))
input = numpy.array(input)
input = input.reshape(1,100)
predictions = modelKeras.predict(input)
print(int(predictions[0]))
