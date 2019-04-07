package com.scratch.ml.supervisedlearning

import java.io.File

import breeze.linalg._
import com.scratch.ml.classify.LogisticRegression
import com.scratch.ml.model.selection.TrainTestSplit

object LogisticRegression {
  def main(args: Array[String]): Unit = {
    //    DenseMatrix.ones[Double](X.rows, 1) / (exp(-X * w) + 1)
    //    val a = DenseMatrix((0.1), (0.2))
    //    println(a)
    //    val b = DenseMatrix.ones[Double](2,1)
    //    println(b/a)
    val root = this.getClass.getClassLoader.getResource("").getFile
    val x = csvread(new File(root + "iris_feature.txt"))
    val y = csvread(new File(root + "iris_type.txt"))
    val (a, b, yTrain, yTest) = new TrainTestSplit().trainTestSplit(x, y)
    val classifier = new LogisticRegression(learningRate = 0.001)
    classifier.fit(a, yTrain)
    println(classifier.w)
    println(classifier.trainingErrors)
    val predict = classifier.predict(b)
    val z = DenseMatrix.horzcat(predict, yTest)
    println(predict :> 0.5)

  }
}
