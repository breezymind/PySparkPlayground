"""
Example of generalized linear regression.
"""

from pyspark.sql import SparkSession
from pyspark.ml.regression import GeneralizedLinearRegression

if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("GeneralizedLinearRegression")\
        .getOrCreate()

    # Load training data
    dataset = spark.read.format("libsvm")\
        .load("data/sample_linear_regression_data.txt")

    glr = GeneralizedLinearRegression(family="gaussian", link="identity",
                                      maxIter=10, regParam=0.3)

    # Fit the model
    model = glr.fit(dataset)

    # Print the coefficients and intercept for generalized linear
    # regression model
    print("Coefficients: " + str(model.coefficients))
    print()
    print("Intercept: " + str(model.intercept))
    print()

    # Summarize the model over the training set and print out some metrics
    summary = model.summary
    print("Coeff. Standard Errors: " + str(summary.coefficientStandardErrors))
    print()
    print("T Values: " + str(summary.tValues))
    print()
    print("P Values: " + str(summary.pValues))
    print()
    print("Dispersion: " + str(summary.dispersion))
    print("Null Deviance: " + str(summary.nullDeviance))
    print("Residual Degree Of Freedom Null: " + str(summary.residualDegreeOfFreedomNull))
    print("Deviance: " + str(summary.deviance))
    print("Residual Degree Of Freedom: " + str(summary.residualDegreeOfFreedom))
    print("AIC: " + str(summary.aic))
    print()
    print("Deviance Residuals: ")
    summary.residuals().show()

    spark.stop()
