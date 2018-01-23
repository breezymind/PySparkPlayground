"""
An example demonstrating Multilayer Perceptron Classifier.

"""

from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql import SparkSession


if __name__ == "__main__":
    spark = SparkSession\
        .builder\
        .appName("MultilayerPerceptronClassifier")\
        .getOrCreate()

    # Loads data.
    dataset = spark.read.format("libsvm")\
        .load("data/sample_multiclass_classification_data.txt")
    dataset.show()

    # Split the data into train and test
    splits = dataset.randomSplit([0.6, 0.4], 42)
    train = splits[0]
    test = splits[1]

    # Specify the layers for the neural network:
    # input layer of size 4 (features), two intermediate of size 5 and 4
    # and output size 3 (classes)
    layers = [4, 5, 4, 3]

    # Create the trainer and set its parameters
    trainer = MultilayerPerceptronClassifier(
        maxIter=100,
        layers=layers,
        blockSize=128,
        seed=42
    )

    # Train the model
    model = trainer.fit(train)

    # Compute accuracy on the test set
    result = model.transform(test)
    predictionAndLabels = result.select("prediction", "label")
    evaluator = MulticlassClassificationEvaluator(metricName="accuracy")
    print("Test set accuracy = " + str(evaluator.evaluate(predictionAndLabels)))

    spark.stop()
