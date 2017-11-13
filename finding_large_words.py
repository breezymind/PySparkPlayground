from pyspark import SparkConf, SparkContext
import os

conf = SparkConf().setMaster("local[*]").setAppName("Finding Large Words")
sc = SparkContext(conf=conf)
sc.setLogLevel("ERROR")

currentDirectory = os.getcwd()
logFile = "file://" + currentDirectory + "/data/log2.txt"
inputRDD = sc.textFile(logFile)

tokenizedRDD = inputRDD.flatMap(lambda x: x.split(" "))
resultsRDD = tokenizedRDD.filter(lambda x: len(x.strip()) > 8)
filteredRDD = resultsRDD.filter(lambda x: "http" not in x)

print(filteredRDD.toDebugString().decode('utf-8'))

for word in filteredRDD.collect():
    print(word, "-->",len(word))

sc.stop()
