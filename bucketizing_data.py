"""
Bucketizing is the transformation of a column of continuous features to a 
column of feature buckets that you define.
"""

from pyspark.sql import SparkSession
from pyspark.ml.feature import Bucketizer

if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("BucketizingData")\
        .getOrCreate()

    # Creating the bucket splits
    splits = [-float("inf"), -0.5, 0.0, 0.5, float("inf")]

    # Creating a simple dataframe
    data = [(-999.9,), (-0.5,), (-0.2,), (0.0,), (0.3,), (999.9,)]
    dataFrame = spark.createDataFrame(data, ["features"])

    bucketizer = Bucketizer(splits=splits, inputCol="features",
                            outputCol="bucketedFeatures")

    # Transform original data into its bucket index.
    bucketedData = bucketizer.transform(dataFrame)

    # Printing result
    print("Bucketizer output with {} buckets".
          format(len(bucketizer.getSplits())-1))
    bucketedData.show()

    spark.stop()
