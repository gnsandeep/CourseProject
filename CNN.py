import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
import json

from numpy import array
from sklearn.model_selection import train_test_split
import seaborn as sns
from numpy import array
from numpy import asarray
from numpy import zeros

from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.layers import Input, Dense, Embedding, Conv2D, MaxPool2D, Conv1D, MaxPooling1D, GlobalMaxPooling1D
from tensorflow.python.keras.layers import Reshape, Flatten, Dropout, Concatenate
from tensorflow.python.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping

#convert train data to DataFrame
def load_training_data_to_pandas(filename = 'data/train.jsonl'):
    X = []
    Y = []
    fhand = open(filename,encoding='utf8')
    for line in fhand:
        data = json.loads(line)

        lt = data['context']
        #lt.reverse()
        fullTweet =   data['response'] + "-" + ''.join(lt)

        X.append(fullTweet)
        Y.append(data['label'])
 
    
    dfdata = pd.DataFrame({'Tweets': X,'Labels': Y}) 

    dfdata.to_csv(r'data/dataPandas.csv',index=False)
    
#Convert test data to DataFrame
def load_test_data_to_pandas(filename = 'data/test.jsonl'):
    tid = []
    X = []
    Y = []
    fhand = open(filename,encoding='utf8')
    for line in fhand:
        data = json.loads(line)
        tid.append(data['id'])
        lt = data['context']
        #lt.reverse()
        fullTweet =   data['response'] + "-" + ''.join(lt)

        X.append(fullTweet)
    
    dftestdata = pd.DataFrame({'ID': tid,'Tweets': X})
    
    dftestdata.to_csv(r'data/dftestdata.csv',index=False)
    
load_training_data_to_pandas()
load_test_data_to_pandas()

twitterdata = pd.read_csv("data/dataPandas.csv")
twitterdata.isnull().values.any()
twitterdata.shape

sns.countplot(x='Labels', data=twitterdata)

#Preprocessing of the data
def preprocess_text(sen):
    # Removing html tags
    sentence = remove_tags(sen)

    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)

    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)

    return sentence

TAG_RE = re.compile(r'<[^>]+>')

def remove_tags(text):
    return TAG_RE.sub('', text)

X = []
sentences = list(twitterdata['Tweets'])
for sen in sentences:
    X.append(preprocess_text(sen))
    
y = twitterdata['Labels']

y = np.array(list(map(lambda x: 1 if x=="SARCASM" else 0, y)))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

#Tokenize the data
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)

X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)
vocab_size = len(tokenizer.word_index)

X_train_tokens = X_train
X_test_tokens = X_test

maxlen = 100
embedding_size = 100

#Word2vec embedding
word2vec = {}
with open('3000tweets_notbinary', encoding='UTF-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vec = np.asarray(values[1:], dtype='float32')
        word2vec[word] = vec
        
num_words = len(list(tokenizer.word_index))
        
embedding_matrix = np.random.uniform(-1, 1, (num_words, embedding_size))
for word, i in tokenizer.word_index.items():
    if i < num_words:
        embedding_vector = word2vec.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
                     
X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)
'''
#Glove embedding
embeddings_dictionary = dict()
glove_file = open('./glove.6B.100d.txt', encoding="utf8")

for line in glove_file:
    records = line.split()
    word = records[0]
    vector_dimensions = asarray(records[1:], dtype='float32')
    embeddings_dictionary [word] = vector_dimensions
glove_file.close()


embedding_matrix = zeros((vocab_size, 100))
for word, index in tokenizer.word_index.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector

#embedding_layer = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=maxlen , trainable=False)     
'''
#Neural Network
model = Sequential()
embedding_layer = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=maxlen , trainable=False)
model.add(embedding_layer)
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

print(model.summary())

num_tokens = [len(tokens) for tokens in X_train_tokens + X_test_tokens]
num_tokens = np.array(num_tokens)

max_tokens = np.mean(num_tokens) + 2 * np.std(num_tokens)
max_tokens = int(max_tokens)
max_tokens

num_classes = 2

#Training params
batch_size = 64 
num_epochs = 25

#Model parameters
num_filters = 64  # görüntünün boyutu mesela 512*512
embed_dim = embedding_size 
weight_decay = 1e-4

print("training CNN ...")
model = Sequential()

#Model add word2vec embedding
embedding_layer = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=maxlen , trainable=False)
model.add(embedding_layer)

#CNN
model.add(Conv1D(num_filters, 7, activation='relu', padding='same'))
model.add(MaxPooling1D(2))
model.add(Conv1D(num_filters, 7, activation='relu', padding='same'))
model.add(GlobalMaxPooling1D())
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l2(weight_decay)))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])

#define callbacks
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=4, verbose=1)
callbacks_list = [early_stopping]

x_train_pad = pad_sequences(X_train_tokens, maxlen=max_tokens)
x_test_pad = pad_sequences(X_test_tokens, maxlen=max_tokens)
y_train2 = []
y_test2 = []

for i in y_train:
  y_train2.append(int(i))
for i in y_test:
  y_test2.append(int(i))
  
y_train2 = np.array(y_train2) 
y_test2 = np.array(y_test2) 
#history = model.fit(x_train_pad, y_train2, batch_size=batch_size, epochs=num_epochs, callbacks=callbacks_list, validation_split=0.1, shuffle=True, verbose=1)
y_train = np.array(y_train)

history = model.fit(X_train, y_train, batch_size=128, epochs=25, verbose=1, validation_split=0.2)

twittertestdata = pd.read_csv("data/dftestdata.csv")
twittertestdata.isnull().values.any()

X_val = []
sentences = list(twittertestdata['Tweets'])
for sen in sentences:
    X_val.append(preprocess_text(sen))
    
X_validate = tokenizer.texts_to_sequences(X_val)
X_valTokens = pad_sequences(X_validate, padding='post', maxlen=maxlen)

X_valTokens = np.array(X_valTokens)

#Predict the model on test data
validation = model.predict(X_valTokens)
twittertestdata['Predict'] = validation

twittertestdata['PLabel'] = np.where(twittertestdata['Predict'] > 0.5, "SARCASM", "NOT_SARCASM")
twittertestdata.head()
twittertestdata.to_csv('answer_CNN.txt', columns = ["ID" , "PLabel"] , header = False , index = False)
