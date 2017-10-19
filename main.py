from pyspark import SparkContext, SparkConf
import utilities

# Creo il SparkContext
conf = SparkConf().setAppName("Project").setMaster("local")
sc = SparkContext(conf=conf)

# Carico i file per la challenge B che servono al machine learning scremandoli di quelli non disponibili
t2015test = utilities.deleteNotAvailable(utilities.loadTsv("tweets_subtask_BD/twitter-2015test-BD-down.tsv", sc))
t2015train = utilities.deleteNotAvailable(utilities.loadTsv("tweets_subtask_BD/twitter-2015train-BD-down.tsv", sc))
t2016dev = utilities.deleteNotAvailable(utilities.loadTsv("tweets_subtask_BD/twitter-2016dev-BD-down.tsv", sc))
t2016devtest = utilities.deleteNotAvailable(utilities.loadTsv("tweets_subtask_BD/twitter-2016devtest-BD-down.tsv", sc))
t2016test = utilities.deleteNotAvailable(utilities.loadTsv("tweets_subtask_BD/twitter-2016test-BD-down.tsv", sc))
t2016train = utilities.deleteNotAvailable(utilities.loadTsv("tweets_subtask_BD/twitter-2016train-BD-down.tsv", sc))


