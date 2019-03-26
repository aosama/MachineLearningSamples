package org.ibrahim.ezmachinelearning

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{DecisionTreeClassificationModel, DecisionTreeClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorAssembler}
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.{DataFrame, functions}
import vegas.sparkExt._
import vegas.{AggOps, Line, Quantitative, Vegas}
import org.apache.spark.ml.linalg.Vector
import vegas._

object DTCensusIncomeExample extends SharedSparkContext {

  def main(args: Array[String]): Unit = {
    val fields = Seq(
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
    val categoricalFieldIndexes = Seq(1, 3, 5, 6, 7, 8, 9, 13)
    val continuousFieldIndexes = Seq(0, 2, 4, 10, 11, 12)

    // Create dataframes to hold census income data
    // Data retrieved from http://archive.ics.uci.edu/ml/datasets/Census+Income
    var trainingData = spark.read.format("csv").load("src/main/resources/adult.data")
    var testData = spark.read.format("csv").load("src/main/resources/adult.test")

    // Format the data
    trainingData = formatData(trainingData, fields, continuousFieldIndexes)
    testData = formatData(testData, fields, continuousFieldIndexes)

    trainingData.printSchema()

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
      .setMaxBins(42) // Since feature "native-country" contains 42 distinct values, need to increase max bins.

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
    showCategories(indexedData, fields, categoricalFieldIndexes)

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

    // Print out the tree with actual column names for features
    var treeModelString = treeModel.toDebugString

    val featureFieldIndexes = categoricalFieldIndexes ++ continuousFieldIndexes
    for (i <- featureFieldIndexes.indices)
      treeModelString = treeModelString
        .replace("feature " + i + " ", fields(featureFieldIndexes(i)) + " ")

    println(s"Learned classification tree model:\n $treeModelString")

    val vectorElem = functions.udf((x: Vector, i: Integer) => x(i))
    val predictionsExpanded = predictions
      .where("indexedLabel = prediction")
      .withColumn("rawPrediction0", vectorElem(predictions.col("rawPrediction"), functions.lit(0)))
      .withColumn("rawPrediction1", vectorElem(predictions.col("rawPrediction"), functions.lit(1)))
      .withColumn("score0", vectorElem(predictions.col("probability"), functions.lit(0)))
      .withColumn("score1", vectorElem(predictions.col("probability"), functions.lit(1)))

    Vegas("Age and Income" , width=Option.apply(800d), height=Option.apply(500d))
      .withDataFrame(predictionsExpanded)
      .mark(Line)
      .encodeX("age", Ordinal)
      .encodeY("score1", Quantitative, aggregate = AggOps.Average)
      .show
  }

  def formatData(df: DataFrame, fields: Seq[String], continuousFieldIndexes: Seq[Int]): DataFrame = {
    var data = df

    // Trim leading spaces from data
    for (colName <- data.columns)
      data = data.withColumn(colName, functions.ltrim(functions.col(colName)))

    // Assign column names
    for (i <- fields.indices)
      data = data.withColumnRenamed("_c" + i, fields(i))

    data = data.withColumnRenamed("_c14", "label")

    // Convert continuous values from string to double
    for (i <- continuousFieldIndexes) {
      data = data.withColumn(fields(i), functions.col(fields(i)).cast("double"))
    }

    // Remove '.' character from label
    data = data.withColumn("label", functions.regexp_replace(functions.col("label"), "\\.", ""))

    data
  }

  def showCategories(df: DataFrame, fields: Seq[String], categoricalFieldIndexes: Seq[Int]): Unit = {
    for (i <- categoricalFieldIndexes) {
      val colName = fields(i)
      df.select(colName + "Indexed", colName).distinct().sort(colName + "Indexed").show(100)
    }
  }
}
