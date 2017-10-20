from pyspark import SparkContext, SparkConf
from pyspark.sql import SQLContext, HiveContext, SparkSession
from pyspark.sql import functions as fn
import utilities

# Configurazioni varie
conf = SparkConf().setAppName("Project").setMaster("local")
sc = SparkContext(conf=conf)
sqlContext = HiveContext(sc)
spark = SparkSession(sc)

# t2015train = loadTsv("/home/fede/PycharmProjects/SentimentAnalysis/tweets_subtask_BD/twitter-2015train-BD-down.tsv")

# Carico i file per la challenge B che servono al machine learning scremandoli di quelli non disponibili
t2015train = utilities.loadTsv("tweets_subtask_BD/twitter-2015train-BD-down.tsv", sc)

# Creo i data frame dal file di training
train_df = t2015train.toDF(['tweet_id', 'topic', 'classification', 'tweet_text'])
train_df.printSchema()

# https://twitter.com/statuses/