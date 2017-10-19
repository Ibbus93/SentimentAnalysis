from pyspark import SparkContext, SparkConf

def deleteNotAvailable(tweets):
    av = []
    for tweet in tweets:
        if tweet[3] != 'Not Available':
            av.append(tweet)
    return av

def loadTsv(file, sc):
    return sc.textFile(file)\
    .map(lambda x : x.encode("ascii", "ignore"))\
    .map(lambda line : line.split("\t"))\
    .collect()