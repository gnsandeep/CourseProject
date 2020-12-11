import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
import json

#nltk.download('stopwords')
from nltk.corpus import stopwords

from nltk.stem import SnowballStemmer,PorterStemmer
from nltk.tokenize import TweetTokenizer

from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer 
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, f1_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix, roc_auc_score, recall_score, precision_score

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

#convert train data to DataFrame
def load_training_data_to_pandas(filename = 'data/train.jsonl'):
    X = []
    Y = []
    fhand = open(filename,encoding='utf8')
    for line in fhand:
        data = json.loads(line)

        fullTweet = data['response']
        X.append(fullTweet)
        Y.append(data['label'])

    dfdata = pd.DataFrame({'Tweets': X,'Labels': Y}) 

    dfdata.to_csv(r'data/dataPandas1.csv',index=False)
    
#Convert test data to DataFrame
def load_test_data_to_pandas(filename = 'data/test.jsonl'):
    tid = []
    X = []
    Y = []
    fhand = open(filename,encoding='utf8')
    for line in fhand:
        data = json.loads(line)
        tid.append(data['id'])

        fullTweet = data['response']
        X.append(fullTweet)

    dftestdata = pd.DataFrame({'ID': tid,'Tweets': X})
    
    dftestdata.to_csv(r'data/dftestdata1.csv',index=False)
    
load_training_data_to_pandas()
load_test_data_to_pandas()

data = pd.read_csv("data/dataPandas1.csv")
data.isnull().values.any()
data_clean = data.copy()
data_clean['Labels'] = data_clean['Labels'].apply(lambda x: 1 if x=='SARCASM' else 0)
data_clean['text_clean'] = data_clean['Tweets'].apply(lambda x: BeautifulSoup(x, "lxml").text)

data_clean = data_clean.loc[:, ['text_clean', 'Labels']]
data_clean.head()

train, test = train_test_split(data_clean, test_size=0.2, random_state=42)
X_train = train['text_clean'].values
X_test = test['text_clean'].values
y_train = train['Labels']
y_test = test['Labels']

#tokenize the data
def tokenize(text): 
    tknzr = TweetTokenizer()
    return tknzr.tokenize(text)

en_stopwords = set(stopwords.words("english")) 

vectorizer = CountVectorizer(
    analyzer = 'word',
    tokenizer = tokenize,
    lowercase = True,
    ngram_range=(1, 1),
    stop_words = en_stopwords)
    
#cross validation and grid search to find good hyperparameters for our SVM model
kfolds = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

np.random.seed(1)

pipeline_svm = make_pipeline(vectorizer,SVC(probability=True, kernel="linear", class_weight="balanced"))

grid_svm = GridSearchCV(pipeline_svm,
                    param_grid = {'svc__C': [0.01, 0.1, 1]}, 
                    cv = kfolds,
                    scoring="roc_auc",
                    verbose=1,   
                    n_jobs=-1) 

grid_svm.fit(X_train, y_train)

model = grid_svm.best_estimator_

X_valTokens = twittertestdata['Tweets'].values

#Predict model on test data set
validation = model.predict(X_valTokens)
twittertestdata['Predict'] = validation
twittertestdata['PLabel'] = np.where(twittertestdata['Predict'] > 0.5, "SARCASM", "NOT_SARCASM")
twittertestdata.head()
twittertestdata.to_csv('answer_SVM.txt', columns = ["ID" , "PLabel"] , header = False , index = False)
