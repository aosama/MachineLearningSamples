package org.ibrahim.ezmachinelearning

import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.{DecisionTreeClassificationModel, DecisionTreeClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, StringIndexer, VectorAssembler}
import org.apache.spark.sql.DataFrame

object DecisionTreeShapeTypeWithCategoricalFeaturesExample extends SharedSparkContext {

  import sqlImplicits._

  def main(args: Array[String]): Unit = {

    val data: DataFrame = Seq(
      (9,"BabyChair"),
      (10,"BabyChair"),
      (11,"BabyChair"),
      (12,"BabyChair"),
      (16,"chair"),
      (17,"chair"),
      (18,"chair"),
      (19,"chair"),
      (18,"table"),
      (29,"table"),
      (30,"table"),
      (31,"table"),
      (29,"table"),
      (30,"table"),
      (31,"table")).toDF("height","shape")

    // Index labels, adding metadata to the label column.
    // Fit on whole dataset to include all labels in index.
    val labelIndexer = new StringIndexer()
      .setInputCol("shape")
      .setOutputCol("indexedLabel")
      .fit(data)

    //    //Categorical Features
    //    val categoricalFeatures = Seq("equalsides")
    //
    //    val categoricalFeaturesIndexer = new StringIndexer()
    //      .setInputCol("equalsides")
    //      .setOutputCol("indexequalsides")
    //
    //    val categoricalFeaturesEncoder = new OneHotEncoderEstimator()
    //      .setInputCols(Seq("indexequalsides").toArray)
    //      .setOutputCols(Seq("onehotequalsides").toArray)


    //Continous Features
    val continousFeatures = Seq("height")


    val featureAssembler = new VectorAssembler()
      .setInputCols(continousFeatures.toArray)
      .setOutputCol("features")

    // Split the data into training and test sets (30% held out for testing).
    val Array(trainingData, testData) = data.randomSplit(Array(0.8, 0.2))

    // Train a DecisionTree model.
    val dt = new DecisionTreeClassifier()
      .setLabelCol("indexedLabel")
      .setFeaturesCol("features")

    // Convert indexed labels back to original labels.
    val labelConverter = new IndexToString()
      .setInputCol("prediction")
      .setOutputCol("predictedLabel")
      .setLabels(labelIndexer.labels)

    // Chain indexers and tree in a Pipeline.
    val pipeline = new Pipeline()
      .setStages(Array(labelIndexer,featureAssembler, dt, labelConverter))

    // Train model. This also runs the indexers.
    val model: PipelineModel = pipeline.fit(trainingData)


    // Make predictions.
    val predictions = model.transform(testData)

    // Select example rows to display.
    predictions.select("predictedLabel", "indexedLabel", "features").show(5)

    // Select (prediction, true label) and compute test error.
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("indexedLabel")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val accuracy = evaluator.evaluate(predictions)
    println(s"Test Error = ${(1.0 - accuracy)}")

    val treeModel = model.stages(2).asInstanceOf[DecisionTreeClassificationModel]
    println(s"Learned classification tree model:\n ${treeModel.toDebugString}")
    // $example off$

    spark.stop()
  }
}
