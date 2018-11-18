/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.finra.msd.basetestclasses

// scalastyle:off
import java.io.File

import org.apache.spark.internal.Logging
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Row, SQLContext, SQLImplicits}
import org.ibrahim.ezmachinelearning.SharedSparkContext
import org.scalatest._

class SparkFunSuite
  extends FunSuite
    with BeforeAndAfterAll
    with Logging
    with Matchers
    with SharedSparkContext
    with ParallelTestExecution {

  protected val outputDirectory: String = System.getProperty("user.dir") + "/sparkOutputDirectory"



  implicit class SequenceImprovements(seq: Seq[Row]) {
    def toDf(schema: StructType): DataFrame = {
      val rowRdd = spark.sparkContext.parallelize(seq)
      val df = spark.createDataFrame(rowRdd, schema)
      return df
    }
  }

  override def beforeAll(): Unit = synchronized {
    super.beforeAll()
  }

  override def afterAll(): Unit = {
    super.afterAll()
  }

  // helper function
  protected final def getTestResourceFile(file: String): File = {
    new File(getClass.getClassLoader.getResource(file).getFile)
  }

  protected final def getTestResourcePath(file: String): String = {
    getTestResourceFile(file).getCanonicalPath
  }


}