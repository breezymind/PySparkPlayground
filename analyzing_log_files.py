from pyspark import SparkConf, SparkContext
import os

conf = SparkConf().setMaster("local[*]").setAppName("Log File Analyzer")
sc = SparkContext(conf=conf)
sc.setLogLevel("ERROR")

def containsError(s):
    return "ERROR" in s

currentDirectory = os.getcwd()
logFile = "file://" + currentDirectory + "/data/log1.txt"

print(logFile)

inputRDD = sc.textFile(logFile)
errorsRDD = inputRDD.filter(containsError)

print("The logfile has " + str(errorsRDD.count()) + " lines with a ERROR response.")
print("Here are the lines:")
for line in errorsRDD.collect():
    print(line)

sc.stop()




