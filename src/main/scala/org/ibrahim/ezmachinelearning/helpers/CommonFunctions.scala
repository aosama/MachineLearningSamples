package org.ibrahim.ezmachinelearning.helpers

import org.apache.spark.ml.PipelineModel
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Row, functions}
import org.ibrahim.ezmachinelearning.DTCensusIncomeExample.spark

object CommonFunctions {
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

  def removeFields(fields: Seq[String], categoricalFieldIndexes: Seq[Int], continuousFieldIndexes: Seq[Int], removeFields: String*):
  (Seq[String], Seq[Int], Seq[Int]) = {
    var fieldsUpdated = fields
    var categoricalFieldIndexesUpdated = categoricalFieldIndexes
    var continuousFieldIndexesUpdated = continuousFieldIndexes

    for (removeField <- removeFields) {
      val removeIndex = fieldsUpdated.indexOf(removeField)
      fieldsUpdated = fieldsUpdated.filter(x => !x.equals(removeField))
      categoricalFieldIndexesUpdated = updateIndex(removeIndex, categoricalFieldIndexesUpdated)
      continuousFieldIndexesUpdated = updateIndex(removeIndex, continuousFieldIndexesUpdated)
    }

    (fieldsUpdated, categoricalFieldIndexesUpdated, continuousFieldIndexesUpdated)
  }

  def updateIndex(removeIndex: Int, indexes: Seq[Int]): Seq[Int] = {
    indexes
      .filter(x => !x.equals(removeIndex))
      .map(x =>
        if (x > removeIndex) x - 1
        else x
      )
  }

  def showCategories(df: DataFrame, fields: Seq[String], categoricalFieldIndexes: Seq[Int], maxRows: Int): Unit = {
    for (i <- categoricalFieldIndexes) {
      val colName = fields(i)
      df.select(colName + "Indexed", colName).distinct().sort(colName + "Indexed").show(maxRows)
    }
  }

  def predictionsForPartialDependencePlot(schema: StructType, indexedData: DataFrame, testData: DataFrame, model: PipelineModel, fieldName: String): DataFrame = {
    var predictions = spark.createDataFrame(spark.sparkContext.emptyRDD[Row], schema)

    for (i <- indexedData.select(fieldName).distinct().orderBy(fieldName).collect()) {
      val testDataIteration = testData.withColumn(fieldName, functions.lit(i.get(0)))
      predictions = predictions.union(model.transform(testDataIteration))
    }

    predictions
  }

  def expandPredictions(predictions: DataFrame): DataFrame = {
    val vectorElem = functions.udf((x: Vector, i: Integer) => x(i))

    predictions
      .where("indexedLabel = prediction")
      .withColumn("rawPrediction0", vectorElem(predictions.col("rawPrediction"), functions.lit(0)))
      .withColumn("rawPrediction1", vectorElem(predictions.col("rawPrediction"), functions.lit(1)))
      .withColumn("score0", vectorElem(predictions.col("probability"), functions.lit(0)))
      .withColumn("score1", vectorElem(predictions.col("probability"), functions.lit(1)))
  }
}
