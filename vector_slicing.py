"""
Example to show how vector slicing works.

This class takes a feature vector and outputs a new feature vector with a 
subarray of the original features.

The subset of features can be specified with either indices (setIndices()) or 
names (setNames()). At least one feature must be selected. Duplicate features 
are not allowed, so there can be no overlap between selected indices and names.

The output vector will order features with the selected indices first (in the 
order given), followed by the selected names (in the order given).
"""

from pyspark.ml.feature import VectorSlicer
from pyspark.ml.linalg import Vectors
from pyspark.sql.types import Row
from pyspark.sql import SparkSession

if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("VectorSlicing")\
        .getOrCreate()

    df = spark.createDataFrame([
        Row(userFeatures=Vectors.sparse(3, {0: -2.0, 1: 2.3})),
        Row(userFeatures=Vectors.dense([-2.0, 2.3, 0.0]))])

    slicer = VectorSlicer(inputCol="userFeatures",
                          outputCol="features",
                          indices=[1])

    output = slicer.transform(df)

    output.select("userFeatures", "features").show()

    spark.stop()
