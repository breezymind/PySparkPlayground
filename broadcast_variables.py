from pyspark import SparkConf, SparkContext

print("Starting...")

conf = SparkConf().setMaster("local[*]").setAppName("BroadcastVariables")
sc = SparkContext(conf=conf)
sc.setLogLevel("ERROR")

broadcastVar = sc.broadcast([1,2,3])
print(broadcastVar.value)
