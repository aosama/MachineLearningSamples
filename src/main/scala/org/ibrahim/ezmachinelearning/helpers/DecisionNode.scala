package org.ibrahim.ezmachinelearning.helpers

case class DecisionNode(featureIndex:Option[Int],
                        gain:Option[Double],
                        impurity:Double,
                        threshold:Option[Double], // Continuous split
                        nodeType:String, // Internal or leaf
                        splitType: Option[String], // Continuous and categorical
                        leftCategories: Option[Array[Double]], // Categorical Split
                        rightCategories: Option[Array[Double]], // Categorical Split
                        prediction:Double,
                        leftChild:Option[DecisionNode],
                        rightChild:Option[DecisionNode])