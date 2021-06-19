import matplotlib.pyplot as plt
import os
import re
import shutil
import string
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import preprocessing
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import pandas as pd
df = pd.read_csv('./spam_or_not_spam/spam_or_not_spam.csv', encoding='utf8')
df.head()

print(df[:4])
