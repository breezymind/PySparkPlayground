"""
An example demonstrating bisecting k-means clustering.

Bisecting k-means is like a combination of k-means and hierarchical clustering.
It starts with all objects in a single cluster and splits clusters in two until
the desired number of clusters in reached.

The critical part is which cluster to choose for splitting. There are different
ways to proceed, for example, you can choose the biggest cluster or the cluster
with the worst quality or a combination of both.

The algorithm starts from a single cluster that contains all points. Iteratively it finds divisible clusters on the bottom level and bisects each of them using k-means, until there are k leaf clusters in total or no leaf clusters are divisible. The bisecting steps of clusters on the same level are grouped together to increase parallelism. If bisecting all divisible clusters on the bottom level would result more than k leaf clusters, larger clusters get higher priority.


"""

from pyspark.ml.clustering import BisectingKMeans
from pyspark.sql import SparkSession


if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("BisectingKMeansExample")\
        .getOrCreate()

    # Loads data.
    dataset = spark.read.format("libsvm").load("data/sample_kmeans_data.txt")
    dataset.show()

    # Training a k-means model.
    bkm = BisectingKMeans().setK(2).setSeed(1)
    model = bkm.fit(dataset)

    # Evaluate clustering by computing Within Set Sum of Squared Errors.
    wssse = model.computeCost(dataset)
    print("Within Set Sum of Squared Errors = " + str(wssse))

    # Shows the result.
    centers = model.clusterCenters()
    print("Cluster Centers: ")
    for center in centers:
        print(center)

    spark.stop()
