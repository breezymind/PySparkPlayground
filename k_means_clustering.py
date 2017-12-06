"""
The K-means algorithm written from scratch against PySpark. In practice,
one may prefer to use the KMeans algorithm of Spark ML.

This example requires NumPy (http://www.numpy.org/).
"""
import sys
import os
import numpy as np
from pyspark.sql import SparkSession


def parseVector(line):
    "Parsing line of data into vector."
    return np.array([float(x) for x in line.split(' ')])


def closestPoint(p, centers):
    """Find the closest point to the centers."""
    bestIndex = 0
    closest = float("+inf")
    for i in range(len(centers)):
        tempDist = np.sum((p - centers[i]) ** 2)
        if tempDist < closest:
            closest = tempDist
            bestIndex = i
    return bestIndex


if __name__ == "__main__":

    # if len(sys.argv) != 3:
    #     print("Usage: kmeans <k> <convergeDist>", file=sys.stderr)
    #     exit(-1)

    print("""WARN: This is a naive implementation of KMeans Clustering and is 
      given as an example! Please refer to kmeans_example.py for an example on 
      how to use ML's KMeans implementation.""", file=sys.stderr)

    spark = SparkSession\
        .builder\
        .appName("KMeansClustering")\
        .getOrCreate()

    currentDirectory = os.getcwd()
    dataFile = "file://" + currentDirectory + "/data/kmeans_data.txt"

    # Loading data into an RDD
    lines = spark.read.text(dataFile).rdd.map(lambda r: r[0])
    # Parse data in vectors and cache the RDD
    data = lines.map(parseVector).cache()

    # K = int(sys.argv[1])
    K = 4
    # convergeDist = float(sys.argv[2])
    convergeDist = 0.01

    kPoints = data.takeSample(False, K, 1)
    tempDist = 1.0

    while tempDist > convergeDist:
        closest = data.map(
            lambda p: (closestPoint(p, kPoints), (p, 1)))
        pointStats = closest.reduceByKey(
            lambda p1_c1, p2_c2: (p1_c1[0] + p2_c2[0], p1_c1[1] + p2_c2[1]))
        newPoints = pointStats.map(
            lambda st: (st[0], st[1][0] / st[1][1])).collect()

        tempDist = sum(np.sum((kPoints[iK] - p) ** 2) for (iK, p) in newPoints)

        for (iK, p) in newPoints:
            kPoints[iK] = p

    print("Final centers: " + str(kPoints))

    spark.stop()