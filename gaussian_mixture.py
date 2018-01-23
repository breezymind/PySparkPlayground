"""
A simple example demonstrating Gaussian Mixture Model (GMM).
"""

from pyspark.ml.clustering import GaussianMixture
from pyspark.sql import SparkSession

if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("GaussianMixture")\
        .getOrCreate()

    # Loading data
    dataset = spark.read.format("libsvm").load("data/sample_kmeans_data.txt")

    # Creating Gaussian Mixture Model
    gmm = GaussianMixture().setK(2).setSeed(42)
    model = gmm.fit(dataset)

    print("Gaussians shown as a DataFrame: ")
    model.gaussiansDF.show(truncate=False)

    spark.stop()
