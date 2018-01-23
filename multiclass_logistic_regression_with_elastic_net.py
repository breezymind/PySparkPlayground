"""
An example demonstrating Multiclass Logistic Regression with ElasticNet.

"""

from pyspark.ml.classification import LogisticRegression
from pyspark.sql import SparkSession


if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("MulticlassLogisticRegressionWithElasticNet")\
        .getOrCreate()

    # Loads data.
    dataset = spark.read.format("libsvm")\
        .load("data/sample_multiclass_classification_data.txt")
    dataset.show()

    lr = LogisticRegression(maxIter=3, regParam=0.3, elasticNetParam=0.8)

    # Fit the model
    lrModel = lr.fit(dataset)

    # Print the coefficients and intercept for multinomial logistic regression
    print("Coefficients: \n" + str(lrModel.coefficientMatrix))
    print("Intercept: " + str(lrModel.interceptVector))

    spark.stop()
