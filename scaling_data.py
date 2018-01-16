"""
Example of how to standardizes features by removing the mean and scaling to 
unit variance.

RBF kernel of Support Vector Machines or the L1 and L2 regularized linear models
typically work better when all features have unit variance and/or zero mean.

Standardization can improve the convergence rate during the optimization
process, and also prevents against features with very large variances exerting
an overly large influence during model training.
"""

from pyspark.ml.feature import StandardScaler
from pyspark.sql import SparkSession

if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("ScalingData")\
        .getOrCreate()

    dataFrame = spark.read.format("libsvm").load("data/sample_libsvm_data.txt")

    # Standardizing features by scaling to unit variance
    scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures",
                            withStd=True, withMean=False)

    # Compute summary statistics by fitting the StandardScaler
    scalerModel = scaler.fit(dataFrame)

    # Normalize each feature to have unit standard deviation.
    scaledData = scalerModel.transform(dataFrame)
    scaledData.show(5)

    # Standardizing features by scaling to unit variance and removing the mean
    scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures",
                            withStd=True, withMean=True)

    # Compute summary statistics by fitting the StandardScaler
    scalerModel = scaler.fit(dataFrame)

    # Normalize each feature to have unit standard deviation.
    scaledData = scalerModel.transform(dataFrame)
    scaledData.show(5)

    spark.stop()
