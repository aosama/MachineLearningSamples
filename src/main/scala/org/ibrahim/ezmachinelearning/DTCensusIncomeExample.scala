package org.ibrahim.ezmachinelearning

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{DecisionTreeClassificationModel, DecisionTreeClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorAssembler}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.ibrahim.ezmachinelearning.helpers.CommonFunctions._
import vegas.sparkExt._
import vegas.{AggOps, Line, Quantitative, Vegas}
import vegas._

object DTCensusIncomeExample extends SharedSparkContext {

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
    val dt = new DecisionTreeClassifier()
      .setLabelCol("indexedLabel")
      .setFeaturesCol("features")
      .setMaxBins(42) // Since feature "native-country" contains 42 distinct values, need to increase max bins to at least 42.
      .setMaxDepth(5)
      .setImpurity("gini")

    // Create object to convert indexed labels back to actual labels for predictions
    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(labelIndexer.labels)

    // Array of stages to run in pipeline
    val indexerArray = Array(labelIndexer) ++ categoricalIndexerArray
    val stageArray = indexerArray ++ Array(vectorAssembler, dt, labelConverter)

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
    println(s"Test error = ${1.0 - accuracy}\n")

    val metrics = new MulticlassMetrics(
      predictions.select("indexedLabel", "prediction")
      .rdd.map(x => (x.getDouble(0), x.getDouble(1)))
    )

    println(s"Confusion matrix:\n ${metrics.confusionMatrix}\n")

    val treeModel = model.stages(stageArray.length - 2).asInstanceOf[DecisionTreeClassificationModel]

    val featureImportances = treeModel.featureImportances.toArray.zipWithIndex.map(x => Tuple2(fields(x._2), x._1)).sortWith(_._2 > _._2)
    println("Feature importances sorted:")
    featureImportances.foreach(x => println(x._1 + ": " + x._2))
    println()

    // Print out the tree with actual column names for features
    var treeModelString = treeModel.toDebugString

    val featureFieldIndexes = categoricalFieldIndexes ++ continuousFieldIndexes
    for (i <- featureFieldIndexes.indices)
      treeModelString = treeModelString
        .replace("feature " + i + " ", fields(featureFieldIndexes(i)) + " ")

    println(s"Learned classification tree model:\n $treeModelString")

    predictions.select("label", Seq("predictedLabel" ,"indexedLabel", "prediction") ++ fields:_*)
      .show()
    val wrongPredictions = predictions
      .select("label", Seq("predictedLabel" ,"indexedLabel", "prediction") ++ fields:_*)
      .where("indexedLabel != prediction")
    wrongPredictions.show()

    // Show the label and all the categorical features mapped to indexes
    val indexedData = new Pipeline()
      .setStages(indexerArray)
      .fit(trainingData)
      .transform(trainingData)
    indexedData.select("indexedLabel", "label").distinct().sort("indexedLabel").show()
    showCategories(indexedData, fields, categoricalFieldIndexes, 100)

    // Partial dependence plots
    val predictionsEducation = predictionsForPartialDependencePlot(predictions.schema, indexedData, testData, model, "education")
    val predictionsMaritalStatus = predictionsForPartialDependencePlot(predictions.schema, indexedData, testData, model, "marital-status")

    val predictionsEducationExpanded = expandPredictions(predictionsEducation)
    val predictionsMaritalStatusExpanded = expandPredictions(predictionsMaritalStatus)

    Vegas("Education and Income" , width=Option.apply(800d), height=Option.apply(500d))
      .withDataFrame(predictionsEducationExpanded)
      .mark(Line)
      .encodeX("education", Ordinal)
      .encodeY("score1", Quantitative, aggregate = AggOps.Average)
      .show

    Vegas("Marital Status and Income" , width=Option.apply(800d), height=Option.apply(500d))
      .withDataFrame(predictionsMaritalStatusExpanded)
      .mark(Line)
      .encodeX("marital-status", Ordinal)
      .encodeY("score1", Quantitative, aggregate = AggOps.Average)
      .show
  }
}
