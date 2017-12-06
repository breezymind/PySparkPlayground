"""
Example of how to standardizes features by removing the mean and scaling to 
unit variance.
"""

from pyspark.ml.feature import StandardScaler
from pyspark.sql import SparkSession

if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("ScalingData")\
        .getOrCreate()

    dataFrame = spark.read.format("libsvm").load("data/sample_libsvm_data.txt")
    scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures",
                            withStd=True, withMean=False)

    # Compute summary statistics by fitting the StandardScaler
    scalerModel = scaler.fit(dataFrame)

    # Normalize each feature to have unit standard deviation.
    scaledData = scalerModel.transform(dataFrame)
    scaledData.show()

    # This time withMean = True
    scaler = StandardScaler(inputCol="features", outputCol="scaledFeatures",
                            withStd=True, withMean=True)

    # Compute summary statistics by fitting the StandardScaler
    scalerModel = scaler.fit(dataFrame)

    # Normalize each feature to have unit standard deviation.
    scaledData = scalerModel.transform(dataFrame)
    scaledData.show()

    spark.stop()
