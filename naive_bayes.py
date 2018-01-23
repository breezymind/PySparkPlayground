"""
Naive Bayes.

Naive Bayes classifiers are a family of simple probabilistic classifiers based 
on applying Bayes’ theorem with strong (naive) independence assumptions between 
the features. The spark.ml implementation currently supports both multinomial 
naive Bayes and Bernoulli naive Bayes.
"""

from pyspark.ml.classification import NaiveBayes
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import SparkSession

if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("NaiveBayes")\
        .getOrCreate()

    # Load training data
    data = spark.read.format("libsvm").load("data/sample_libsvm_data.txt")

    # Split the data into train and test
    splits = data.randomSplit([0.6, 0.4], seed=42)
    train = splits[0]
    test = splits[1]

    # create the trainer and set its parameters
    nb = NaiveBayes(smoothing=1.0, modelType="multinomial")

    # train the model
    model = nb.fit(train)

    # select example rows to display.
    predictions = model.transform(test)
    predictions.show()

    # compute accuracy on the test set
    evaluator = MulticlassClassificationEvaluator(labelCol="label",
                                                  predictionCol="prediction",
                                                  metricName="accuracy")
    accuracy = evaluator.evaluate(predictions)
    print("Test set accuracy = " + str(accuracy))

    spark.stop()
