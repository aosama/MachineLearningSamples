package org.ibrahim.ezmachinelearning

import org.apache.spark.SparkConf
import org.apache.spark.sql._

trait SharedSparkContext {


  var sparkSession: SparkSession = {
    val conf = new SparkConf().setAppName("megasparkdiff")
      .setMaster("local[*]")
      .set("spark.driver.host", "localhost")
      .set("spark.ui.enabled", "true") //disable spark UI
    sparkSession = SparkSession.builder.config(conf).getOrCreate()
    sparkSession.sparkContext.setLogLevel("warn")
    sparkSession
  }


  implicit class DataFrameImprovements(df: DataFrame) {
    def getColumnsSeq(): Seq[Column] = {
      val s: Seq[Column] = df.columns.map(c => df(c)).toSeq
      return s
    }
  }

  protected object sqlImplicits extends SQLImplicits {
    protected override def _sqlContext: SQLContext = sparkSession.sqlContext
  }

}
