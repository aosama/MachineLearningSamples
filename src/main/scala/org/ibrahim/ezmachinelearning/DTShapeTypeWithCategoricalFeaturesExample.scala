package org.ibrahim.ezmachinelearning

import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.classification.{DecisionTreeClassificationModel, DecisionTreeClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{IndexToString, OneHotEncoderEstimator, StringIndexer, VectorAssembler}
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.types.DoubleType
import org.ibrahim.ezmachinelearning.helpers.SparkMLTree

object DTShapeTypeWithCategoricalFeaturesExample extends SharedSparkContext {

  import sqlImplicits._

  def main(args: Array[String]): Unit = {

    val data: DataFrame = Seq(
      ("small","BabyChair" ,1),
      ("small","BabyChair",2),
      ("small","BabyChair",3),
      ("small","BabyChair",4),
      ("fullSize","chair",5),
      ("fullSize","chair",6),
      ("fullSize","chair",7),
      ("fullSize","chair",8),
      ("fullSize","chair",9),
      ("n/a","table",10),
      ("n/a","table",11),
      ("n/a","table",12),
      ("n/a","table",13),
      ("n/a","table",14),
      ("n/a","table",15)).toDF("handRest","shape" , "recNumber")

    // split train test
    val fractions = Map("BabyChair" -> 0.8, "chair" -> 0.8, "table" -> 0.8)
    val trainingData = data.stat.sampleBy("shape" , fractions, 100l)
    val testData = data.except(trainingData)

    trainingData.show()
    testData.show()
    // Index labels, adding metadata to the label column.
    // Fit on whole dataset to include all labels in index.
    val labelIndexer = new StringIndexer()
      .setInputCol("shape")
      .setOutputCol("indexedLabel")
      .fit(data)

    //Categorical Features
    val categoricalFeaturesIndexer = new StringIndexer()
      .setInputCol("handRest")
      .setOutputCol("indexedhandRest")

    val categoricalFeaturesEncoder = new OneHotEncoderEstimator()
      .setInputCols(Seq("indexedhandRest").toArray)
      .setOutputCols(Seq("VecHandRest").toArray)

    //Feature Assembler
    val featureAssembler = new VectorAssembler()
      .setInputCols(Seq("VecHandRest").toArray)
      .setOutputCol("features")

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
      .setStages(Array(
        labelIndexer,
        categoricalFeaturesIndexer,
        categoricalFeaturesEncoder,
        featureAssembler,
        dt,
        labelConverter))

    // Train model. This also runs the indexers.
    val model: PipelineModel = pipeline.fit(trainingData)

    // Make predictions.
    val predictions = model.transform(testData)

    // Select (prediction, true label) and compute test error.
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("indexedLabel")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")
    val accuracy = evaluator.evaluate(predictions)
    println(s"Test Error = ${(1.0 - accuracy)}")

    val treeModel = model.stages(4).asInstanceOf[DecisionTreeClassificationModel]
    println(s"Learned classification tree model:\n ${treeModel.toDebugString}")
    // $example off$
    predictions.show(false)
    spark.stop()

    val tree_json = new SparkMLTree(treeModel).toJsonPlotFormat()
    println(tree_json)
  }
}
