"""
Alternating Least Squares Matrix Factorization Example.
"""

from pyspark.sql import SparkSession
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row

if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("ALSSparkML")\
        .getOrCreate()

    # Parsing the data and creating a dataframe
    lines = spark.read.text("data/sample_movielens_ratings.txt").rdd
    parts = lines.map(lambda row: row.value.split("::"))
    ratingsRDD = parts.map(lambda p: Row(userId=int(p[0]), movieId=int(p[1]),
                                       rating=float(p[2]), timestamp=int(p[3])))
    ratings = spark.createDataFrame(ratingsRDD)

    # Splitting the data in training and test set
    (training, test) = ratings.randomSplit([0.8, 0.2])

    # Build the recommendation model using ALS on the training data
    # Cold start strategy = 'drop' to ensure we don't get NaN evaluation
    # metrics in case of new users.
    als = ALS(maxIter=5, regParam=0.01, userCol="userId", itemCol="movieId",
              ratingCol="rating", coldStartStrategy="drop")
    model = als.fit(training)

    # Evaluate the model by computing the RMSE on the test data
    predictions = model.transform(test)
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                    predictionCol="prediction")
    rmse = evaluator.evaluate(predictions)
    print("Root-mean-square error = " + str(rmse))

    # Generate top 10 movie recommendations for each user
    userRecs = model.recommendForAllUsers(10)
    # Generate top 10 user recommendations for each movie
    movieRecs = model.recommendForAllItems(10)
    userRecs.show(truncate=False)
    movieRecs.show(truncate=False)

    spark.stop()
