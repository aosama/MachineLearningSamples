package org.ibrahim.ezmachinelearning

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{DecisionTreeClassificationModel, DecisionTreeClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorAssembler}
import org.apache.spark.sql.functions

object DecisionTreeCensusIncome extends SharedSparkContext {

  def main(args: Array[String]): Unit = {
    val categoricalFields = Seq(1, 3, 5, 6, 7, 8, 9, 13)
    val continuousFields = Seq(2, 4, 10, 11, 12)

    // Create dataframe to hold census income data
    // Data retrieved from http://archive.ics.uci.edu/ml/datasets/Census+Income
    var data = spark.read.format("csv").load("src/main/resources/adult.test.txt")

    // Create object to convert categorical values to index values
    val categoricalIndexerArray = for (i <- categoricalFields)
      yield new StringIndexer()
        .setInputCol("_c" + i)
        .setOutputCol("_c" + i + "Indexed")

    // Convert continuous values from string to double
    for (i <- continuousFields) {
      data = data.withColumn("_c" + i, functions.col("_c" + i).cast("double"))
    }

    // Create object to index label values
    val labelIndexer = new StringIndexer()
      .setInputCol("_c14")
      .setOutputCol("indexedLabel")
      .fit(data)

    // Create object to generate feature vector from categorical and continuous values
    val vectorAssembler = new VectorAssembler()
      .setInputCols((categoricalFields.map(x => "_c" + x + "Indexed") ++ continuousFields.map(x => "_c" + x)).toArray)
      .setOutputCol("features")

    // Split data into training and test data
    val Array(trainingData, testData) = data.randomSplit(Array(0.7, 0.3))

    val dt = new DecisionTreeClassifier()
      .setLabelCol("indexedLabel")
      .setFeaturesCol("features")
      .setMaxBins(41) // Since feature _c13 contains 41 distinct values, need to increase max bins.

    // Create object to convert indexed labels back to actual labels for predictions
    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(labelIndexer.labels)

    // Array of stages to run in pipeline
    val stageArray = Array(labelIndexer) ++ categoricalIndexerArray ++ Array(vectorAssembler, dt, labelConverter)

    val pipeline = new Pipeline()
      .setStages(stageArray)

    // Train the model
    val model = pipeline.fit(trainingData)

    // Test the model
    val predictions = model.transform(testData)

    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("indexedLabel")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")

    val accuracy = evaluator.evaluate(predictions)
    println(s"Test error = ${(1.0 - accuracy)}")

    val treeModel = model.stages(stageArray.length - 2).asInstanceOf[DecisionTreeClassificationModel]
    println(s"Learned classification tree model:\n ${treeModel.toDebugString}")
  }
}
