package com.example.analysis

import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql.functions.{avg, col, desc, expr, round, udf, stddev_samp}

import scala.math.sqrt

object StockPriceAnalysis extends App{

  val spark = SparkUtilities.createSpark("analysisProject")

  val stockDF = SparkUtilities.readCSV("./src/resources/sourceCSV/stock_prices.csv", spark)

  stockDF.printSchema()
  stockDF.describe().show()
  stockDF.show(5, false)

  /**
   * User defined Function to help calculate daily return
   *
   * To calculate your daily return as a percentage:
   * subtract the opening price from the closing price.
   * Then, divide the outcome by the opening price.
   * Finally, multiply the outcome by 100 to convert to a percentage
   *
   * @param openingPrice Double
   * @param closingPrice Double
   * @return Double
   */
  def dailyReturn(openingPrice:Double, closingPrice:Double):Double = (closingPrice - openingPrice)/openingPrice *100
  spark.udf.register("return", dailyReturn(_:Double, _:Double):Double)
  val dailyReturnUdf = udf(dailyReturn(_:Double, _:Double):Double)

  //add the daily return as a new column
  val dfWithReturn = stockDF
    .withColumn("dailyReturn", dailyReturnUdf(col("open"), col("close")))

  //aggregate average daily return
  val result = dfWithReturn
    .groupBy("date")
    .agg(avg("dailyReturn").alias("averageReturn"))
    .orderBy(desc("date"))

  result.show(10, false)
  //Save the results to the file as Parquet(CSV optional)
  SparkUtilities.saveParquet(result, "./src/resources/analysis/parquet/stocksWithAvgReturn")
  SparkUtilities.saveCSV(result, "./src/resources/analysis/csv/stocksWithAvgReturn")

  //Which stock was traded most frequently - as measured by closing price * volume - on average?

  //new column with the frequency for each ticker each day
  val tradingFreqDF = stockDF
    .withColumn("trading", col("close") * col("volume"))

  //find the average for each ticker
 val tradingResult = tradingFreqDF
    .groupBy("ticker")
    .agg(avg("trading").alias("averageTrading"))
    .orderBy(round(col("averageTrading"), 2).desc)//.orderBy(desc("averageTrading"))

  tradingResult.show(false)
  SparkUtilities.saveParquet(tradingResult, "./src/resources/analysis/parquet/mostTradedAvg")
  SparkUtilities.saveCSV(tradingResult, "./src/resources/analysis/csv/mostTradedAvg")


  //Which stock was the most volatile as measured by annualized standard deviation of daily returns?
  //Annualized Standard Deviation = Standard Deviation of Daily Returns * Square Root (250)

 val standDev2016 = dfWithReturn
   .where("date LIKE '2016-%'")
   .groupBy("ticker")
   .agg(stddev_samp("dailyReturn").alias("StandDev_dailyReturn"))
   .withColumn("AnnualizedStandardDeviation_for_2016", round(col("StandDev_dailyReturn") * sqrt(250), 2))
   .orderBy(desc("AnnualizedStandardDeviation_for_2016"))
   .show(false)

  val standDev2015 = dfWithReturn
    .where("date LIKE '2015-%'")
    .groupBy("ticker")
    .agg(stddev_samp("dailyReturn").alias("StandDev_dailyReturn"))
    .withColumn("AnnualizedStandardDeviation_for_2015", round(col("StandDev_dailyReturn") * sqrt(250), 2))
    .orderBy(desc("AnnualizedStandardDeviation_for_2015"))
    .show(false)


  val dfWithPreviousDay = stockDF.withColumn("prevHigh", expr("" +
    "LAG (high,1,0) " +
    "OVER (PARTITION BY ticker " +
    "ORDER BY date )"))
    .withColumn("prevLow", expr("" +
      "LAG (low,1,0) " +
      "OVER (PARTITION BY ticker " +
      "ORDER BY date )"))
  dfWithPreviousDay.show(10, false)

  import org.apache.spark.ml.feature.RFormula
  val supervised = new RFormula()
    .setFormula("high ~ prevHigh + prevLow ")

  val ndf = supervised
    .fit(dfWithPreviousDay) //prepares the formula
    .transform(dfWithPreviousDay) //generally transform will create the new data


    ndf.show(10, false)

    val cleanDf = ndf.where("prevHigh != 0.0")
    cleanDf.show(10, false)

//  val linReg = new LinearRegression()
//
//  val Array(train,test) = cleanDf.randomSplit(Array(0.75,0.25))
//
//  val lrModel = linReg.fit(train)
//
//  val intercept = lrModel.intercept
//  val coefficients = lrModel.coefficients
//  val x1 = coefficients(0)
//  val x2 = coefficients(1)
//
//  println(s"Intercept: $intercept and coefficient for x1 is $x1 and for x2 is $x2")
//
//  val summary = lrModel.summary
//
//  //to truly test this model we should be using different stocks or different dates for these 3 stocks
//
//  val predictedDf = lrModel.transform(test)
//
//  predictedDf.show(10, false)

  import org.apache.spark.ml.Pipeline
  import org.apache.spark.ml.evaluation.RegressionEvaluator
  import org.apache.spark.ml.feature.VectorIndexer
  import org.apache.spark.ml.regression.DecisionTreeRegressionModel
  import org.apache.spark.ml.regression.DecisionTreeRegressor


  // Automatically identify categorical features, and index them.
  // Here, we treat features with > 4 distinct values as continuous.
  val featureIndexer = new VectorIndexer()
    .setInputCol("features")
    .setOutputCol("indexedFeatures")
    .setMaxCategories(2)
    .fit(cleanDf)

  // Split the data into training and test sets (30% held out for testing).
  val Array(trainingData, testData) = cleanDf.randomSplit(Array(0.75, 0.25))

  // Train a DecisionTree model.
  val dt = new DecisionTreeRegressor()
    .setLabelCol("label")
    .setFeaturesCol("indexedFeatures")

  // Chain indexer and tree in a Pipeline.
  val pipeline = new Pipeline()
    .setStages(Array(featureIndexer, dt))

  // Train model. This also runs the indexer.
  val model = pipeline.fit(trainingData)

  // Make predictions.
  val predictions = model.transform(testData)

  // Select example rows to display.
  predictions.select("prediction", "label", "features").show(5)

  // Select (prediction, true label) and compute test error.
  val evaluator = new RegressionEvaluator()
    .setLabelCol("label")
    .setPredictionCol("prediction")
    .setMetricName("rmse")
  val rmse = evaluator.evaluate(predictions)
  println(s"Root Mean Squared Error (RMSE) on test data = $rmse")

  val treeModel = model.stages(1).asInstanceOf[DecisionTreeRegressionModel]
  println(s"Learned regression tree model:\n ${treeModel.toDebugString}")

}
