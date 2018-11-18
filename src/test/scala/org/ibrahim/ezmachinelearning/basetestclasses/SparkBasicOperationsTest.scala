package org.ibrahim.ezmachinelearning.basetestclasses

import org.finra.msd.basetestclasses.SparkFunSuite

class SparkBasicOperationsTest extends SparkFunSuite{

  import sqlImplicits._

  test("basic spark operations")  {
    val df = Seq(
      ("1", "1" , "Adam" ,"Andreson"),
      ("2","2","Bob","Branson"),
      ("4","4","Chad","Charly"),
      ("5","5","Joe","Smith"),
      ("5","5","Joe","Smith"),
      ("6","6","Edward","Eddy"),
      ("7","7","normal","normal")
    ).toDF("key1" , "key2" , "value1" , "value2")

    assert(df.count() == 7)
  }

}
