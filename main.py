import pandas as pd
import numpy as np

from string import digits

from pyspark import SparkContext, SparkConf
from pyspark.sql import HiveContext, SparkSession, SQLContext, functions as fn
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import StopWordsRemover, CountVectorizer, RegexTokenizer, Tokenizer, IDF, VectorAssembler, PCA
from pyspark.ml import Pipeline

import importlib, utilities, requests, itertools, time, re


conf = SparkConf().setAppName("Project").setMaster("local").set('spark.executor.memory', '8G').set('spark.driver.memory', '45G').set('spark.driver.maxResultSize', '10G')
sc = SparkContext(conf=conf)

sqlContext = SQLContext(sc)
spark = SparkSession(sc)

importlib.reload(utilities)


# ## 1. Caricamento set di train e test e relative key

# ### Train

df_train = pd.read_csv("new_train_df.tsv", sep='\t', encoding='utf-8', names=['tweet_id', 'topic', 'class', 'tweet_text', 'categories', 'concepts'])
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


# ### Test

df_test = pd.read_csv("test_BD.tsv", sep='\t', encoding='utf-8')
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


# ## 4. Creazione delle feature e delle pipeline

stop_words = requests.get('http://ir.dcs.gla.ac.uk/resources/linguistic_utils/stop_words').text.split()

tokenizer1 = Tokenizer().setInputCol('tweet_text').setOutputCol('words')
sw_filter1 = StopWordsRemover().setStopWords(stop_words).setCaseSensitive(False).setInputCol("words").setOutputCol("filtered")
cve1 = CountVectorizer(minTF=1., minDF=5., vocabSize=2**17).setInputCol("filtered").setOutputCol("tf")
idf1 = IDF().setInputCol('tf').setOutputCol('tfidf')

tokenizer2 = Tokenizer().setInputCol('tweet_cat').setOutputCol('words_cat')
sw_filter2 = StopWordsRemover().setStopWords(stop_words).setCaseSensitive(False).setInputCol("words_cat").setOutputCol("filtered_cat")
cve2 = CountVectorizer(minTF=1., minDF=5., vocabSize=2**17).setInputCol("filtered_cat").setOutputCol("tf_cat")
idf2 = IDF().setInputCol('tf_cat').setOutputCol('tfidf_cat')

tokenizer3 = Tokenizer().setInputCol('tweet_con').setOutputCol('words_con')
sw_filter3 = StopWordsRemover().setStopWords(stop_words).setCaseSensitive(False).setInputCol("words_con").setOutputCol("filtered_con")
cve3 = CountVectorizer(minTF=1., minDF=5., vocabSize=2**17).setInputCol("filtered_con").setOutputCol("tf_con")
idf3 = IDF().setInputCol('tf_con').setOutputCol('tfidf_con')

vecAss = VectorAssembler(inputCols=['tfidf', 'tfidf_cat', 'tfidf_con'], outputCol='features')

pipe_feat1 = Pipeline(stages=[tokenizer1, sw_filter1, cve1, idf1])
pipe_feat2 = Pipeline(stages=[tokenizer2, sw_filter2, cve2, idf2])
pipe_feat3 = Pipeline(stages=[tokenizer3, sw_filter3, cve3, idf3])

lr = LogisticRegression().setLabelCol('class').setFeaturesCol('features').setRegParam(0.0).setMaxIter(100).setElasticNetParam(0.)


# ## Esecuzione Task B

avg_rec = []
f1_score = []
accuracy = []

i = 0

for topic in np.unique(df_train.topic):    
    index_test = [x for x in  utilities.getSimilar(df_train, df_test, topic)[0]] 
    
    topic_train = df_train[df_train.topic == topic]
    topic_test = df_test.iloc[index_test]
    
    df_f = sqlContext.createDataFrame(pd.concat([topic_train, topic_test]))

    tr1 = pipe_feat1.fit(df_f).transform(df_f)    
    tr2 = pipe_feat2.fit(tr1).transform(tr1)
    tr3 = pipe_feat3.fit(tr2).transform(tr2)

    tr3_test = vecAss.transform(tr3).select('topic', 'features', 'class')
                    
    train_topic = tr3_test.filter(tr3_test['topic'] == topic)
    test_topic = tr3_test.filter(tr3_test['topic'] != topic)    

    pred = lr.fit(train_topic).transform(test_topic)
    tp, tn, fp, fn = utilities.getConfusionMatrix(pred)
    
    try:
        p_pos = tp / (tp + fp)
    except ZeroDivisionError:
        p_pos = 0
        
    try:
        p_neg = tn / (tn + fn)
    except ZeroDivisionError:
        p_neg = 0
        
    acc_aux = (tp + tn) / (tp + tn + fp + fn)
    f1_aux = (2*tp) / ((2*tp) + fp + fn)
    avg_rec_aux = ( p_pos + p_neg ) / 2
    
    accuracy.append(acc_aux)
    f1_score.append(f1_aux)
    avg_rec.append(avg_rec_aux)
    
    if i % 100 == 0:
        print("Topic processati:", i)
    
    i = i + 1
    
print("Accuracy:", np.mean(accuracy))
print("F1 Score:", np.mean(f1_score))
print("Avg Rec:", np.mean(avg_rec))


