"""
This example demonstrates applying TrainValidationSplit to split data
and preform model selection.
"""

from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import LinearRegression
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.sql import SparkSession


if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("TrainValidationSplit")\
        .getOrCreate()

    # Prepare training and test data.
    data = spark.read.format("libsvm")\
        .load("data/sample_linear_regression_data.txt")
    train, test = data.randomSplit([0.9, 0.1], seed=12345)

    # Creating a Linear Regression model
    lr = LinearRegression(maxIter=10)

    # We use a ParamGridBuilder to construct a grid of parameters to search over.
    # TrainValidationSplit will try all combinations of values and determine
    # best model using the evaluator.
    paramGrid = ParamGridBuilder()\
        .addGrid(lr.regParam, [0.1, 0.01]) \
        .addGrid(lr.fitIntercept, [False, True])\
        .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0])\
        .build()

    # In this case the estimator is simply the linear regression.
    # A TrainValidationSplit requires an Estimator, a set of Estimator
    # ParamMaps, and an Evaluator.
    # 80% of the data will be used for training, 20% for validation.
    tvs = TrainValidationSplit(estimator=lr,
                               estimatorParamMaps=paramGrid,
                               evaluator=RegressionEvaluator(),
                               trainRatio=0.8)

    # Run TrainValidationSplit, and choose the best set of parameters.
    model = tvs.fit(train)

    # Make predictions on test data. model is the model with combination of parameters
    # that performed best.
    model.transform(test)\
        .select("features", "label", "prediction")\
        .show()

    spark.stop()
