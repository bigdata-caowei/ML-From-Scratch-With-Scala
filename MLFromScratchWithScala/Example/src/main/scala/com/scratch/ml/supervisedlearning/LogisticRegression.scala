package com.scratch.ml.supervisedlearning

import java.io.File

import breeze.linalg._
import com.scratch.ml.classify.LogisticRegression
import com.scratch.ml.model.preprocess.Normalization
import com.scratch.ml.model.selection.{Evaluator, TrainTestSplit}

object LogisticRegression {
  def main(args: Array[String]): Unit = {
    //读取数据
    val root = this.getClass.getClassLoader.getResource("").getFile
    val x = Normalization.normalize(csvread(new File(root + "iris_feature.txt")))
    val y = csvread(new File(root + "iris_type.txt"))

    //拆分测试训练数据集
    val (xTrain, xTest, yTrain, yTest) = TrainTestSplit.split(x, y, shuffle = true)


    //构造逻辑回归分类器
    val classifier = new LogisticRegression(learningRate = 0.01, nIterations = 100000)
    println("*" * 95)

    //训练
    classifier.fit(xTrain, yTrain)
    println("*" * 88)

    //评价测试样本的准确率
    val predict = classifier.predict(xTest)
    val accuracy = Evaluator.accuracy(yTest :> 0.5, predict)

    println(accuracy)
  }
}
