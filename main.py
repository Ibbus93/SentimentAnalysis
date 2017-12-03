from pyspark import SparkContext, SparkConf
from pyspark.sql import HiveContext, SparkSession
from pyspark.sql import SQLContext
from pyspark.ml.feature import StopWordsRemover
from pyspark.ml.feature import CountVectorizer
from pyspark.ml.feature import RegexTokenizer
from pyspark.ml import Pipeline
from pyspark.ml.feature import IDF

import requests
import utilities

# Configurazioni varie
conf = SparkConf().setAppName("Project").setMaster("local")
sc = SparkContext(conf=conf)
sqlContext = SQLContext(sc)
spark = SparkSession(sc)

# t2015train = loadTsv("/home/fede/PycharmProjects/SentimentAnalysis/tweets_subtask_BD/twitter-2015train-BD-down.tsv")

# Carico i file per la challenge B che servono al machine learning scremandoli di quelli non disponibili
t2015train = utilities.loadTsv("tweets_subtask_BD/twitter-2015train-BD-down.tsv", sc)


for x in t2015train.collect():
    print(x)

# Creo i data frame dal file di training
train_df = t2015train.toDF(['tweet_id', 'topic', 'classification', 'tweet_text'])
train_df.printSchema()

print(train_df.show(10))

# Carico le stop words e le rimuovo
stop_words = requests.get('http://ir.dcs.gla.ac.uk/resources/linguistic_utils/stop_words').text.split()
stop_words = [x.encode('UTF8') for x in stop_words]


sw_filter = StopWordsRemover()\
  .setStopWords(stop_words)\
  .setCaseSensitive(False)\
  .setInputCol("words")\
  .setOutputCol("filtered")

# Computo il IDF su tutte le parole

cv = CountVectorizer(minTF=1., minDF=5., vocabSize=2**17)\
  .setInputCol("filtered")\
  .setOutputCol("tf")

tokenizer = RegexTokenizer().setGaps(False)\
  .setPattern("\\p{L}+")\
  .setInputCol("tweet_text")\
  .setOutputCol("words")

review_words_df = tokenizer.transform(train_df)

cv_pipeline = Pipeline(stages=[tokenizer, sw_filter, cv]).fit(train_df)
print(cv_pipeline.transform(train_df).show(5))

idf = IDF().\
    setInputCol('tf').\
    setOutputCol('tfidf')

idf_pipeline = Pipeline(stages=[cv_pipeline, idf]).fit(train_df)
print(idf_pipeline.transform(train_df).show(5))
