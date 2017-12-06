"""
An example demonstrating Logistic Regression 
and Multinomial Logistic Regression.
"""

from pyspark.ml.classification import LogisticRegression
from pyspark.sql import SparkSession


if __name__ == "__main__":
    spark = SparkSession \
        .builder \
        .appName("LogisticRegression") \
        .getOrCreate()

    # Load training data
    training = spark.read.format("libsvm").load("data/sample_libsvm_data.txt")

    training.show()

    # Applying Logistic Regression algorithm with regularizaton parameter = 0.3
    # and elastic net regularization parameter = 0.8
    lr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8)

    # If you need an explanation of a parameter, you can use the explainParam
    # method which gives you name, doc, default and current value.
    print(lr.explainParam("elasticNetParam"))

    # Fit the model
    lrModel = lr.fit(training)

    # Print the coefficients and intercept for logistic regression
    print("Coefficients: " + str(lrModel.coefficients))
    print("Intercept: " + str(lrModel.intercept))

    # Extract the summary of training results from the returned
    # LogisticRegressionModel instance
    trainingSummary = lrModel.summary

    # Obtain the objective (= scaled loss + regularization) per iteration
    objectiveHistory = trainingSummary.objectiveHistory
    print("\nobjectiveHistory:")
    for objective in objectiveHistory:
        print(objective)

    # Obtain the receiver-operating characteristic as a dataframe and
    # areaUnderROC.
    trainingSummary.roc.show()
    print("areaUnderROC: " + str(trainingSummary.areaUnderROC))

    # Set the model threshold to maximize F-Measure
    fMeasure = trainingSummary.fMeasureByThreshold

    maxFMeasure = fMeasure.groupBy()\
        .max('F-Measure') \
        .select('max(F-Measure)') \
        .head()

    bestThreshold = fMeasure \
        .where(fMeasure['F-Measure'] == maxFMeasure['max(F-Measure)']) \
        .select('threshold').head()['threshold']

    print("Best threshold: ", bestThreshold)
    print()
    lr.setThreshold(bestThreshold)

    # We can also use the multinomial family for binary classification
    mlr = LogisticRegression(maxIter=10, regParam=0.3, elasticNetParam=0.8,
                             family="multinomial")

    # Fit the model
    mlrModel = mlr.fit(training)

    # Print the coefficients and intercepts for logistic regression with multinomial family
    print("Multinomial coefficients: " + str(mlrModel.coefficientMatrix))
    print("Multinomial intercepts: " + str(mlrModel.interceptVector))

    spark.stop()
