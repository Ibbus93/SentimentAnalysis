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
t2015test = utilities.loadTsv("tweets_subtask_BD/twitter-2015test-BD-down.tsv", sc)
t2015train = utilities.loadTsv("tweets_subtask_BD/twitter-2015train-BD-down.tsv", sc)
t2016dev = utilities.loadTsv("tweets_subtask_BD/twitter-2016dev-BD-down.tsv", sc)
t2016devtest = utilities.loadTsv("tweets_subtask_BD/twitter-2016devtest-BD-down.tsv", sc)
t2016test = utilities.loadTsv("tweets_subtask_BD/twitter-2016test-BD-down.tsv", sc)
t2016train = utilities.loadTsv("tweets_subtask_BD/twitter-2016train-BD-down.tsv", sc)

print t2015test.take(5)

# https://twitter.com/statuses/