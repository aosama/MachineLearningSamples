// Databricks notebook source
val data = Seq(
  (9,"BabyChair"),
  (10,"BabyChair"),
  (11,"BabyChair"),
  (12,"BabyChair"),
  (16,"chair"),
  (17,"chair"),
  (18,"chair"),
  (19,"chair"),
  (18,"chair"),
  (29,"table"),
  (30,"table"),
  (31,"table"),
  (29,"table"),
  (30,"table"),
  (31,"table")).toDF("height","shape")

// COMMAND ----------

// Index labels, adding metadata to the label column.
// Fit on whole dataset to include all labels in index.
import org.apache.spark.ml.feature._
val labelIndexer = new StringIndexer()
  .setInputCol("shape")
  .setOutputCol("indexedLabel")
  .fit(data)

// COMMAND ----------

//Continous Features
val continousFeatures = Seq("height")

// COMMAND ----------

val featureAssembler = new VectorAssembler()
  .setInputCols(continousFeatures.toArray)
  .setOutputCol("features")

// COMMAND ----------

// Split the data into training and test sets (30% held out for testing).
val Array(trainingData, testData) = data.randomSplit(Array(0.8, 0.2))

// COMMAND ----------

// Train a DecisionTree model.
import org.apache.spark.ml.classification.{DecisionTreeClassificationModel, DecisionTreeClassifier}
val dt = new DecisionTreeClassifier()
  .setLabelCol("indexedLabel")
  .setFeaturesCol("features")

// COMMAND ----------

// Convert indexed labels back to original labels.
val labelConverter = new IndexToString()
  .setInputCol("prediction")
  .setOutputCol("predictedLabel")
  .setLabels(labelIndexer.labels)

// COMMAND ----------

// Chain indexers and tree in a Pipeline.
import org.apache.spark.ml.{Pipeline, PipelineModel}
val pipeline = new Pipeline()
  .setStages(Array(labelIndexer,featureAssembler, dt, labelConverter))

// COMMAND ----------

// Train model. This also runs the indexers.
val model: PipelineModel = pipeline.fit(trainingData)

// COMMAND ----------

// Make predictions.
val predictions = model.transform(testData)

// Select example rows to display.
predictions.select("predictedLabel", "indexedLabel", "features").show(5)

// COMMAND ----------

// Select (prediction, true label) and compute test error.
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
val evaluator = new MulticlassClassificationEvaluator()
  .setLabelCol("indexedLabel")
  .setPredictionCol("prediction")
  .setMetricName("accuracy")
val accuracy = evaluator.evaluate(predictions)
println(s"Test Error = ${(1.0 - accuracy)}")

val treeModel = model.stages(2).asInstanceOf[DecisionTreeClassificationModel]
println(s"Learned classification tree model:\n ${treeModel.toDebugString}")

// COMMAND ----------

import org.apache.spark.sql.types.DoubleType
val analysisDataDF = spark.range(0 , 40).toDF("height")
      .withColumn("height" , 'height.cast(DoubleType))

// COMMAND ----------

val opDf = model.transform(analysisDataDF)
display(opDf)

// COMMAND ----------

val tree = model.stages(2).asInstanceOf[DecisionTreeClassificationModel]
display(tree)
