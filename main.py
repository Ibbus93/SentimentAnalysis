from pyspark import SparkContext, SparkConf

conf = SparkConf().setAppName("Project").setMaster("local")
sc = SparkContext(conf=conf)

