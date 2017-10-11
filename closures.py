from pyspark import SparkConf, SparkContext

conf = SparkConf().setMaster("local[*]").setAppName("Closures")
sc = SparkContext(conf=conf)
sc.setLogLevel("ERROR")

counter = 0
data = [1,2,3,4]
rdd = sc.parallelize(data)

# Wrong: Don't do this! Use Accumulators instead!
# Prior to execution, Spark computes the task's closure.
# The closure is those variables and methods which must be visible for the
# executor to perform its computations on the RDD.
# This closure is serialized and sent to each executor, which means that
# counter is now a copy of the counter on the driver node.

def increment_counter(x):
    global counter
    counter += x

rdd.foreach(increment_counter)

print("Counter value:", counter)