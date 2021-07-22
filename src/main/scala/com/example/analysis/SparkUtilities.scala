package com.example.analysis

import org.apache.spark.sql.{DataFrame, SparkSession}

object SparkUtilities extends App{
  /**
   * A method for creating a spark session
   * @param appName
   * @param verbose
   * @param master
   * @return
   */
  def createSpark(appName:String, verbose:Boolean = true, master: String= "local"): SparkSession = {
    if (verbose) println(s"$appName with Scala version: ${util.Properties.versionNumberString}")

    val spark = SparkSession.builder().appName(appName).master(master).getOrCreate()
    spark.conf.set("spark.sql.shuffle.partitions", "5") //recommended for local
    if (verbose) println(s"Session started on Spark version ${spark.version}")
    spark
  }

  /**
   *
   * @param filePath String from where to read the file
   * @param spark SparkSession
   * @return DataFrame
   */
  def readCSV(filePath:String, spark:SparkSession):DataFrame = spark.read
    .format("csv")
    .option("header", true)
    .option("inferSchema", true)
    .option("path", filePath)
    .load

  /**
   * Method that saves the data in parquet format
   * @param df DataFrame
   * @param destPath String
   */
  def saveParquet(df:DataFrame, destPath:String):Unit = {
    df.coalesce(1)
      .write
      .format("parquet")
      .mode("overwrite")
      .save(destPath)
  }

  /**
   * Method that saves the data in CSV format
   * @param df
   * @param destPath
   */
  def saveCSV(df:DataFrame, destPath:String):Unit = {
    df.coalesce(1)
      .write
      .format("csv")
      .option("header", true)
      .mode("overwrite")
      .save(destPath)
  }

}
