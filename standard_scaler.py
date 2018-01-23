"""
Standard Scaler.

Standardizes features by removing the mean and scaling to unit variance using 
column summary statistics on the samples in the training set.

The "unit std" is computed using the corrected sample standard deviation, which 
is computed as the square root of the unbiased sample variance.
"""

from pyspark.ml.feature import StandardScaler
from pyspark.sql import SparkSession

if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("StandardScaler")\
        .getOrCreate()

    dataFrame = spark.read.format("libsvm").load("data/sample_libsvm_data.txt")

    scaler = StandardScaler(inputCol="features",
                            outputCol="scaledFeatures",
                            withStd=True,
                            withMean=False)

    # Compute summary statistics by fitting the StandardScaler
    scalerModel = scaler.fit(dataFrame)

    # Normalize each feature to have unit standard deviation.
    scaledData = scalerModel.transform(dataFrame)
    scaledData.show()

    spark.stop()
