import pandas as pd
import numpy as np

from string import digits

from pyspark import SparkContext, SparkConf
from pyspark.sql import HiveContext, SQLContext, functions as fn
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import StopWordsRemover, CountVectorizer, RegexTokenizer, Tokenizer, IDF, VectorAssembler, PCA
from pyspark.ml import Pipeline

import importlib, utilities, requests, itertools, time, re

conf = SparkConf().setAppName("Project").setMaster("local")
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)

# 1. Caricamento set di train e test e relative key
# Train
print("Caricamento Train...")

df_train = pd.read_csv("train_CE.tsv", sep='\t', encoding='utf-8', names=['tweet_id', 'topic', 'class', 'tweet_text', 'categories', 'concepts'])
df_train['categories'] = df_train['categories'].fillna('').map(lambda s : re.sub('^ ', '', s.lower().replace('/', ' ').replace(';', '').replace(',',''))).map(utilities.clean_tweet)
df_train['concepts'] = df_train['concepts'].fillna('').map(lambda s : s.lower().replace(';', ' ')).map(utilities.clean_tweet)

df_train['tweet_text'] = df_train['tweet_text'].map(utilities.clean_tweet)

tweet_cat = []
tweet_con = []
ttk = []

for tweet, cats, cons in zip(df_train['tweet_text'], df_train['categories'], df_train['concepts']):
    tweet_cat.append(tweet + ' ' + cats)
    tweet_con.append(tweet + ' ' + cons)
    
for topic in np.unique(df_train.topic):
    if df_train[df_train.topic == topic].shape[0] > 20:
        ttk.append(topic)
    
df_train['tweet_cat'] = tweet_cat
df_train['tweet_con'] = tweet_con

df_train = df_train[df_train['topic'].isin(ttk)]


# Test
print("Caricamento Train...")
df_test = pd.read_csv("test_CE.tsv", sep='\t', encoding='utf-8',  names=['tweet_id', 'topic', 'class', 'tweet_text', 'categories', 'concepts'])
df_test['categories'] = df_test['categories'].fillna('').map(lambda s : re.sub('^ ', '', s.lower().replace('/', ' ').replace(';', '').replace(',',''))).map(utilities.clean_tweet)
df_test['concepts'] = df_test['concepts'].fillna('').map(lambda s : s.lower().replace(';', ' ')).map(utilities.clean_tweet)    

df_test['tweet_text'] = df_test['tweet_text'].map(utilities.clean_tweet)

tweet_cat = []
tweet_con = []

for tweet, cats, cons in zip(df_test['tweet_text'], df_test['categories'], df_test['concepts']):
    tweet_cat.append(tweet + ' ' + cats)
    tweet_con.append(tweet + ' ' + cons)
    
df_test['tweet_cat'] = tweet_cat
df_test['tweet_con'] = tweet_con

# 2. Creazione delle feature
stop_words = requests.get('http://ir.dcs.gla.ac.uk/resources/linguistic_utils/stop_words').text.split()

tokenizer2 = Tokenizer().setInputCol('tweet_cat').setOutputCol('words_cat')
sw_filter2 = StopWordsRemover().setStopWords(stop_words).setCaseSensitive(False).setInputCol("words_cat").setOutputCol("filtered_cat")
cve2 = CountVectorizer(minTF=1., minDF=5., vocabSize=2**17).setInputCol("filtered_cat").setOutputCol("tf_cat")
idf2 = IDF().setInputCol('tf_cat').setOutputCol('tfidf_cat')

tokenizer3 = Tokenizer().setInputCol('tweet_con').setOutputCol('words_con')
sw_filter3 = StopWordsRemover().setStopWords(stop_words).setCaseSensitive(False).setInputCol("words_con").setOutputCol("filtered_con")
cve3 = CountVectorizer(minTF=1., minDF=5., vocabSize=2**17).setInputCol("filtered_con").setOutputCol("tf_con")
idf3 = IDF().setInputCol('tf_con').setOutputCol('tfidf_con')

pipe_feat2 = Pipeline(stages=[tokenizer2, sw_filter2, cve2, idf2])
pipe_feat3 = Pipeline(stages=[tokenizer3, sw_filter3, cve3, idf3])

# 3. Esecuzione Task B
# a. Categorie
print("Esecuzione Task C con categorie...")
mae_m, mae_ni = utilities.task_C(df_train, df_test, sc, sqlContext, pipe_feat2, 'tfidf_cat')

print 'Categorie: '
print '  MAE m: %f' % (mae_m)
print '  MAE ni: %f' % (mae_ni)

# b. Concetti
print("Esecuzione Task C con concetti...")
mae_m, mae_ni = utilities.task_C(df_train, df_test, sc, sqlContext, pipe_feat3, 'tfidf_con')

print 'Concetti: '
print '  MAE m: %f' % (mae_m)
print '  MAE ni: %f' % (mae_ni)
