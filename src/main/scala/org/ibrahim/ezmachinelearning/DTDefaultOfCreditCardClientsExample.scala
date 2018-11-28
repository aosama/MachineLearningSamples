//package org.ibrahim.ezmachinelearning
//
//import org.apache.spark.ml.{Pipeline, PipelineModel}
//import org.apache.spark.ml.classification.{DecisionTreeClassificationModel, DecisionTreeClassifier}
//import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
//import org.apache.spark.ml.feature.{IndexToString, OneHotEncoderEstimator, StringIndexer, VectorAssembler}
//import org.apache.spark.sql.DataFrame
//
//object DTDefaultOfCreditCardClientsExample extends SharedSparkContext {
//
//  import sqlImplicits._
//  def main(args: Array[String]): Unit = {
//
//    val data = spark.read.format("csv")
//      .option("header", "true")
//      .option("inferSchema", "true")
//      .load(Configs.defaultOfCreditCardClientsCsv)
//    // split train test
//    val fractions = Map(1 -> 0.8 , 0 -> 0.8)
//    val trainingData = data.stat.sampleBy("default_payment_next_month" , fractions, 100l)
//    val testData = data.except(trainingData)
//
//    println(data.filter('default_payment_next_month === 0).count())
//    println(data.filter('default_payment_next_month === 1).count())
//
//    println(data.count())
//    println(trainingData.count())
//    println(testData.count())
//
//
//    //continous features
//    val continousFeatures = Seq(
//      "LIMIT_BAL",
//      "SEX",
//      "EDUCATION",
//      "MARRIAGE",
//      "AGE",
//      "PAY_0",
//      "PAY_2",
//      "PAY_3",
//      "PAY_4",
//      "PAY_5",
//      "PAY_6",
//      "BILL_AMT1",
//      "BILL_AMT2",
//      "BILL_AMT3",
//      "BILL_AMT4",
//      "BILL_AMT5",
//      "BILL_AMT6",
//      "PAY_AMT1",
//      "PAY_AMT2",
//      "PAY_AMT3",
//      "PAY_AMT4",
//      "PAY_AMT5",
//      "PAY_AMT6").toArray
//
//
//    //Feature Assembler
//    val featureAssembler = new VectorAssembler()
//      .setInputCols(Seq("VecHandRest").toArray)
//      .setOutputCol("features")
//
//    // Train a DecisionTree model.
//    val dt = new DecisionTreeClassifier()
//      .setLabelCol("indexedLabel")
//      .setFeaturesCol("features")
//
//
//
//    // Chain indexers and tree in a Pipeline.
//    val pipeline = new Pipeline()
//      .setStages(Array(
//          featureAssembler,
//        dt))
//
//    // Train model. This also runs the indexers.
//    val model: PipelineModel = pipeline.fit(trainingData)
//
//    // Make predictions.
//    val predictions = model.transform(testData)
//
//    // Select (prediction, true label) and compute test error.
//    val evaluator = new MulticlassClassificationEvaluator()
//      .setLabelCol("indexedLabel")
//      .setPredictionCol("prediction")
//      .setMetricName("accuracy")
//    val accuracy = evaluator.evaluate(predictions)
//    println(s"Test Error = ${(1.0 - accuracy)}")
//
//    val treeModel = model.stages(4).asInstanceOf[DecisionTreeClassificationModel]
//    println(s"Learned classification tree model:\n ${treeModel.toDebugString}")
//    // $example off$
//
//    predictions.show(false)
//    spark.stop()
//
//  }
//}
