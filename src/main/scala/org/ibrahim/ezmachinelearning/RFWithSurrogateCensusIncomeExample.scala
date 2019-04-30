package org.ibrahim.ezmachinelearning

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{DecisionTreeClassificationModel, DecisionTreeClassifier, RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorAssembler}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.functions
import org.ibrahim.ezmachinelearning.helpers.CommonFunctions._

object RFWithSurrogateCensusIncomeExample extends SharedSparkContext {

  def main(args: Array[String]): Unit = {
    var fields = Seq(
      "age",
      "workclass",
      "fnlwgt",
      "education",
      "education-num",
      "marital-status",
      "occupation",
      "relationship",
      "race",
      "sex",
      "capital-gain",
      "capital-loss",
      "hours-per-week",
      "native-country"
    )
    var categoricalFieldIndexes = Seq(1, 3, 5, 6, 7, 8, 9, 13)
    var continuousFieldIndexes = Seq(0, 2, 4, 10, 11, 12)

    // Create dataframes to hold census income data
    // Data retrieved from http://archive.ics.uci.edu/ml/datasets/Census+Income
    var trainingData = spark.read.format("csv").load("src/main/resources/adult.data")
    var testData = spark.read.format("csv").load("src/main/resources/adult.test")

    // Format the data
    trainingData = formatData(trainingData, fields, continuousFieldIndexes)
    testData = formatData(testData, fields, continuousFieldIndexes)

    // Add unique identifier to data for use in surrogate model
    trainingData = trainingData.withColumn("uniqueIndex", functions.monotonically_increasing_id())
    testData = testData.withColumn("uniqueIndex", functions.monotonically_increasing_id())

    // Exclude redundant and weighted attributes from feature vector
    val (fieldsUpdated, categoricalFieldIndexesUpdated, continuousFieldIndexesUpdated) = removeFields(
      fields, categoricalFieldIndexes, continuousFieldIndexes, "education-num", "relationship", "fnlwgt")
    fields = fieldsUpdated
    categoricalFieldIndexes = categoricalFieldIndexesUpdated
    continuousFieldIndexes = continuousFieldIndexesUpdated

    // Create object to convert categorical values to index values
    val categoricalIndexerArray =
      for (i <- categoricalFieldIndexes)
        yield new StringIndexer()
          .setInputCol(fields(i))
          .setOutputCol(fields(i) + "Indexed")

    // Create object to index label values
    val labelIndexer = new StringIndexer()
      .setInputCol("label")
      .setOutputCol("indexedLabel")
      .fit(trainingData)

    // Create object to generate feature vector from categorical and continuous values
    val vectorAssembler = new VectorAssembler()
      .setInputCols((categoricalFieldIndexes.map(i => fields(i) + "Indexed") ++ continuousFieldIndexes.map(i => fields(i))).toArray)
      .setOutputCol("features")

    // Create decision tree
    val rf = new RandomForestClassifier()
      .setLabelCol("indexedLabel")
      .setFeaturesCol("features")
      .setNumTrees(50)
      .setMaxBins(100) // Since feature "native-country" contains 42 distinct values, need to increase max bins to at least 42.
      .setMaxDepth(10)
      .setImpurity("gini")

    // Create object to convert indexed labels back to actual labels for predictions
    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(labelIndexer.labels)

    // Array of stages to run in pipeline
    val indexerArray = Array(labelIndexer) ++ categoricalIndexerArray
    val stageArray = indexerArray ++ Array(vectorAssembler, rf, labelConverter)

    val pipeline = new Pipeline()
      .setStages(stageArray)

    // Train the model
    val model = pipeline.fit(trainingData)

    // Test the model
    val predictionsTest = model.transform(testData)

    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("indexedLabel")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")

    val accuracy = evaluator.evaluate(predictionsTest)
    println(s"Test error = ${1.0 - accuracy}\n")

    val metrics = new MulticlassMetrics(
      predictionsTest.select("indexedLabel", "prediction")
        .rdd.map(x => (x.getDouble(0), x.getDouble(1)))
    )

    println(s"Confusion matrix:\n ${metrics.confusionMatrix}\n")

    val treeModel = model.stages(stageArray.length - 2).asInstanceOf[RandomForestClassificationModel]

    val featureImportances = treeModel.featureImportances.toArray.zipWithIndex.map(x => Tuple2(fields(x._2), x._1)).sortWith(_._2 > _._2)
    println("Feature importances sorted:")
    featureImportances.foreach(x => println(x._1 + ": " + x._2))
    println()

    // Show the label and all the categorical features mapped to indexes
    val indexedData = new Pipeline()
      .setStages(indexerArray)
      .fit(trainingData)
      .transform(trainingData)
    indexedData.select("indexedLabel", "label").distinct().sort("indexedLabel").show()
    showCategories(indexedData, fields, categoricalFieldIndexes, 100)

    // Get random forest predictions on training data for use in training surrogate model
    val predictionsTraining = model.transform(trainingData)

    // Build training and test data set for surrogate based on predictions from random forest
    val trainingDataSurrogate = predictionsTraining.select((Seq("predictedLabel", "uniqueIndex") ++ fields).map(x => functions.col(x)):_*)
      .withColumnRenamed("predictedLabel", "label")
    val testDataSurrogate = predictionsTest.select((Seq("predictedLabel", "uniqueIndex") ++ fields).map(x => functions.col(x)):_*)
      .withColumnRenamed("predictedLabel", "label")

    // Create decision tree
    val dt = new DecisionTreeClassifier()
      .setLabelCol("indexedLabel")
      .setFeaturesCol("features")
      .setMaxBins(42) // Since feature "native-country" contains 42 distinct values, need to increase max bins to at least 42.
      .setMaxDepth(5)
      .setImpurity("gini")

    val stageArraySurrogate = indexerArray ++ Array(vectorAssembler, dt, labelConverter)

    val pipelineSurrogate = new Pipeline()
      .setStages(stageArraySurrogate)

    // Train the surrogate model
    val modelSurrogate = pipelineSurrogate.fit(trainingDataSurrogate)

    // Test the surrogate model
    val predictionsSurrogate = modelSurrogate.transform(testDataSurrogate)

    val accuracySurrogate = evaluator.evaluate(predictionsSurrogate)
    println(s"Test error = ${1.0 - accuracySurrogate}\n")

    println(s"Confusion matrix:\n ${metrics.confusionMatrix}\n")

    val treeModelSurrogate = modelSurrogate.stages(stageArraySurrogate.length - 2).asInstanceOf[DecisionTreeClassificationModel]

    val featureImportancesSurrogate = treeModelSurrogate.featureImportances.toArray.zipWithIndex.map(x => Tuple2(fields(x._2), x._1)).sortWith(_._2 > _._2)
    println("Feature importances sorted:")
    featureImportancesSurrogate.foreach(x => println(x._1 + ": " + x._2))
    println()

    // Print out the tree with actual column names for features
    var treeModelStringSurrogate = treeModelSurrogate.toDebugString

    val featureFieldIndexes = categoricalFieldIndexes ++ continuousFieldIndexes
    for (i <- featureFieldIndexes.indices)
      treeModelStringSurrogate = treeModelStringSurrogate
        .replace("feature " + i + " ", fields(featureFieldIndexes(i)) + " ")

    println(s"Learned classification tree model:\n $treeModelStringSurrogate")

    // Analyze how close the predictions of the surrogate model are to the random forest
    val predictionsJoined = predictionsSurrogate.as("a")
      .join(predictionsTest.as("b"), Seq("uniqueIndex"), "left_outer")
      .selectExpr("a.prediction as predictionSurrogate", "b.prediction")

    spark.udf.register("difference-square",
      (x: Double, y: Double) => scala.math.pow(x - y, 2)
    )

    val meanPrediction = predictionsJoined.agg(functions.mean("prediction")).collect()(0).getDouble(0)

    var sse = predictionsJoined.withColumn("sse", functions.callUDF("difference-square",
      functions.col("predictionSurrogate"),
      functions.col("prediction"))
    ).agg(functions.sum("sse").as("sse")).select("sse").collect()(0).getDouble(0)

    var sst = predictionsJoined.withColumn("sst", functions.callUDF("difference-square",
      functions.col("predictionSurrogate"),
      functions.lit(meanPrediction))
    ).agg(functions.sum("sst").as("sst")).select("sst").collect()(0).getDouble(0)

    println(s"How close are the predictions of the surrogate model to the random forest?\n ${1 - (sse / sst)}\n")
  }
}
