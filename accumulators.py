from pyspark import SparkConf, SparkContext

print("Starting...")

conf = SparkConf().setMaster("local[*]").setAppName("Accumulators")
sc = SparkContext(conf=conf)
sc.setLogLevel("ERROR")

accum = sc.accumulator(0)
print(accum)

sc.parallelize([1,2,3,4]).foreach(lambda x: accum.add(x))
print(accum.value)
