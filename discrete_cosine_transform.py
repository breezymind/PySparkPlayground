"""
Discrete Cosine Transform Example.

A discrete cosine transform (DCT) expresses a finite sequence of data points in 
terms of a sum of cosine functions oscillating at different frequencies.

A DCT is a Fourier-related transform similar to the discrete Fourier transform 
(DFT), but using only real numbers.
"""

from pyspark.ml.feature import DCT
from pyspark.ml.linalg import Vectors
from pyspark.sql import SparkSession

if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("DiscreteCosineTransform")\
        .getOrCreate()

    df = spark.createDataFrame([
        (Vectors.dense([0.0, 1.0, -2.0, 3.0]),),
        (Vectors.dense([-1.0, 2.0, 4.0, -7.0]),),
        (Vectors.dense([14.0, -2.0, -5.0, 1.0]),)], ["features"])

    df.show(truncate=False)

    dct = DCT(inverse=False, inputCol="features", outputCol="featuresDCT")
    dctDf = dct.transform(df)
    dctDf.select("featuresDCT").show(truncate=False)

    # Doing the reverse
    dct_inv = DCT(inverse=True, inputCol="featuresDCT",
                  outputCol="origFeatures").transform(dctDf)
    dct_inv.select("origFeatures").show(truncate=False)

    spark.stop()
