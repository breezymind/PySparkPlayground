"""
Binarization is the process of thresholding numerical features to binary (0/1)
features.
"""

from pyspark.sql import SparkSession
from pyspark.ml.feature import Binarizer

if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("BinarizingData")\
        .getOrCreate()

    # Creating simple dataframe
    continuousDataFrame = spark.createDataFrame([
        (0, 0.21),
        (1, 0.85),
        (2, 0.28),
        (3, 0.50),
        (4, 0.51)
    ], ["id", "feature"])

    # Transforming original data to binarized data with threshold = 0.5
    binarizer = Binarizer(threshold=0.5, inputCol="feature",
                          outputCol="binarized_feature")
    binarizedDataFrame = binarizer.transform(continuousDataFrame)

    # Printing result
    print("Binarized data with threshold = {:.2f}".
          format(binarizer.getThreshold()))
    binarizedDataFrame.show()

    spark.stop()
