from pyspark import SparkContext, SparkConf

conf = SparkConf().setAppName("Project").setMaster("local")
sc = SparkContext(conf=conf)

twitter2015testBD = sc.textFile("tweets_subtask_BD/twitter-2015test-BD-down.tsv").map(lambda line : line.split("\t")).collect()