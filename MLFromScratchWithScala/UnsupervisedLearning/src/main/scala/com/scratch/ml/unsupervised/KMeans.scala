package com.scratch.ml.unsupervised

import java.util.Random

import breeze.linalg.{DenseMatrix, DenseVector, argmin}
import com.scratch.ml.model.debug.Loggable
import com.scratch.ml.model.math.MathHelper

import scala.collection.mutable.ArrayBuffer

class KMeans(val k: Int = 2, val maxIterations: Int = 500) extends Loggable {

  var centroids: DenseMatrix[Double] = _

  val clusters = new Array[ArrayBuffer[DenseVector[Double]]](k)


  def predict(X: DenseMatrix[Double]): Array[ArrayBuffer[DenseVector[Double]]] = {
    initializeCentroids(X)
    for (i <- 0 until maxIterations) {
      createClusters(X)
      val oldCentroids = centroids.copy;
      recalculateCentroids()
    }
    return clusters
  }


  private def initializeCentroids(X: DenseMatrix[Double]): Unit = {
    centroids = DenseMatrix.zeros[Double](k, X.cols)
    var temp: DenseMatrix[Double] = X
    val random = new Random()
    for (i <- 0 until k) {
      val randIndex = random.nextInt(temp.rows)
      val choice = temp(randIndex, ::)
      centroids(i, ::) := choice
      val middle = DenseMatrix.zeros[Double](temp.rows - 1, X.cols)
      middle(0 until randIndex, ::) := temp(0 until randIndex, ::)
      middle(randIndex until middle.rows, ::) := temp(randIndex + 1 until temp.rows, ::)
      temp = middle
    }
    logger.debug("初始化centroids完成 \n{}", centroids)
  }


  private def createClusters(X: DenseMatrix[Double]): Array[ArrayBuffer[DenseVector[Double]]] = {

    for (i <- 0 until k) {
      clusters(i) = new ArrayBuffer[DenseVector[Double]]()
    }

    for (i <- 0 until X.rows) {
      val x = X(i, ::).t
      classifySample(x)
    }

    return clusters
  }

  private def classifySample(x: DenseVector[Double]): Unit = {
    val vector = DenseVector.zeros[Double](k)
    for (i <- 0 until k) {
      vector(i) = MathHelper.euclideanDistance(x, centroids(i, ::).t)
    }

    clusters(argmin(vector)).append(x)

  }

  private def recalculateCentroids(): Unit = {

    for (i <- 0 until centroids.rows) {
      val sum = clusters(i).reduce(_ + _)
      centroids(i, ::) := sum.t / clusters(i).size.toDouble
    }


  }

}
