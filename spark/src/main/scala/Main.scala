import java.io.IOException
import java.util

import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.functions.{abs, col, udf, when}
import org.apache.spark.sql.types.{FloatType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import ucar.nc2.time.CalendarDate
import ucar.nc2.{NetcdfFile, Variable}

import scala.collection.JavaConversions._
import scala.collection.mutable.ListBuffer


object Main {
  private val CALENDAR: String = "proleptic_gregorian"
  private val UNITS: String = "days since 1950-01-01"

  // TODO: read directly as required type
  @throws[IOException]
  private def readVariable(variable: Variable) : Array[Float] = {
    val data: Array[Float] = variable.read.get1DJavaArray(classOf[Float]).asInstanceOf[Array[Float]]

    val scaleFactorAttribute = variable.findAttribute("scale_factor")
    if (scaleFactorAttribute != null) {
      val scaleFactor = scaleFactorAttribute.getNumericValue.floatValue
      var i = 0
      while (i < data.length) {
        data(i) *= scaleFactor
        i += 1
      }
    }
    data
  }

  def cartesian[X, Y, Z](a: Array[X], b: Array[Y], c: Array[Z]): Array[(X, Y, Z)] = {
    for { x <- a; y <- b; z <- c } yield (x, y, z)
  }

  // actually expects 3 dimensions; TODO: generalize
  private def readData(path: String, ss: SparkSession, dims: List[String], createIndex: Boolean, numPartitions: Int): DataFrame = {
    val file: NetcdfFile = NetcdfFile.open(path)
    val vars: util.List[Variable] = file.getVariables
    // split variables into dimensions and regular data
    val dimVars: Map[String, Variable] = vars.filter(v => dims.contains(v.getShortName)).map(v => v.getShortName -> v).toMap
    val colVars: Map[String, Variable] = vars.filter(v => !dims.contains(v.getShortName)).map(v => v.getShortName -> v).toMap

    // prepare for cartesian product
    val lon: Array[Float] = readVariable(dimVars(dims(0)))
    val lat: Array[Float] = readVariable(dimVars(dims(1)))
    val tim: Array[Float] = readVariable(dimVars(dims(2)))
    val dimsCartesian: Array[(Float, Float, Float)] = cartesian(lon, lat, tim)

    // create the rdd with the dimensions (by transposing the cartesian product)
    var tempRDD: RDD[ListBuffer[_]] = ss.sparkContext.parallelize(dimsCartesian.map(t => ListBuffer(t._1, t._2, t._3)), numPartitions)
    // gather the names of the columns (in order)
    val names: ListBuffer[String] = ListBuffer(dims: _*)

    // read the columns and zip with the rdd
    for (col <- colVars) {
      tempRDD = tempRDD.zip(ss.sparkContext.parallelize(readVariable(col._2), numPartitions)).map(t => t._1 ++ Seq(t._2))
      names.add(col._1)
    }

    // add the index column
    if (createIndex) {
      tempRDD = tempRDD.zipWithIndex().map(t => t._1 ++ Seq(t._2.asInstanceOf[Float]))
      names.add("index")
    }

    // create final RDD[Row] and use it to make the DataFrame
    val finalRDD: RDD[Row] = tempRDD.map(Row.fromSeq(_))
    // TODO: case class is probably the best idea here
    val df: DataFrame = ss.createDataFrame(finalRDD, StructType(names.map(StructField(_, FloatType, nullable = false))))

    // convert time from float to String
    val floatTimeToString = udf((time: Float) => {
      val udunits = String.valueOf(time.asInstanceOf[Int]) + " " + UNITS

      CalendarDate.parseUdunits(CALENDAR, udunits).toString.substring(0, 10)
    })
    // return final df
    df.withColumn("time", floatTimeToString(df("time")))
  }

  type ArgsMap = Map[Symbol, Any]
  @throws(classOf[RuntimeException])
  def parseArgs(map : ArgsMap, list: List[String]) : ArgsMap = {
    list match {
      case Nil => map
      case "--input" :: value :: tail =>
        parseArgs(map ++ Map('input -> value), tail)
      case "--partitions" :: value :: tail =>
        parseArgs(map ++ Map('partitions -> value.toInt), tail)
      case "--output" :: value :: tail =>
        parseArgs(map ++ Map('output -> value), tail)
      case "--check" :: tail =>
        parseArgs(map ++ Map('check -> true), tail)
        //stop parsing here and catch in main
      case string => throw new RuntimeException("invalid argument: " + string)
    }
  }

  def main(args: Array[String]): Unit = {
    val validNumberArgs: List[Int] = List(4, 7)
    if (!validNumberArgs.contains(args.length)) {
      throw new RuntimeException("Usage: --input <path> --partitions <number_partitions> or: " +
        "--input <path> --partitions <number_partitions> --output <path> --check")
    }
    val options = parseArgs(Map(), args.toList)

    val spark: SparkSession = SparkSession.builder
      .appName("Spark Pipeline")
      .getOrCreate()

    val dimensions: List[String] = List("longitude", "latitude", "time")
    val df1: DataFrame = readData(options('input) + "data1.nc", spark, dimensions,
                                  createIndex = true, options('partitions).asInstanceOf[Int])
    val df2: DataFrame = readData(options('input) + "data2.nc", spark, dimensions,
                                  createIndex = false, options('partitions).asInstanceOf[Int])

    // PIPELINE
    // 1. join the 2 dataframes
    var df: DataFrame = df1.join(df2, dimensions, "inner").cache()

    // 2. quick preview on the data
    if (options.contains('check)) {
      // this will not actually be the first 10 rows, so don't compare for correctness; to select the actual first 10
      // would require more unnecessary work for spark
      df.limit(10)
        .write
        .option("header", "true")
        .csv(options('output) + "head")
    } else {
      println(df.show(10))
    }

    // 3. subset the data
    // the only way to select by row number; more effectively would be to just filter by latitude as intended, though
    // this deviates even further from the other pipeline implementations
    df = df.filter(df("index") >= 709920.0f && df("index") < 1482480.0f)

    // 4. drop rows with null values
    df = df.filter(df("tg") =!= -99.99f && df("pp") =!= -999.9f && df("rr") =!= -999.9f)

    // 5. drop columns
    df = df.drop("pp_err", "rr_err", "index")

    // 6. UDF 1: compute absolute difference between max and min
    df = df.withColumn("abs_diff", abs(df("tx") - df("tn"))).cache()

    // 7. explore the data through aggregations
    val df_agg = df.drop("longitude", "latitude", "time")
      .summary("min", "max", "mean", "stddev")
    if (options.contains('check)) {
      df_agg.withColumnRenamed("summary", "agg")
        .withColumn("agg", when(col("agg") === "stddev", "std").otherwise(col("agg")))
        .write
        .option("header", "true")
        .csv(options('output) + "agg")
    } else {
      println(df_agg.show())
    }

    // 8. compute mean per month
    // UDF 2: compute custom year+month format
    val computeYearMonth = udf((time: String) => {
      time.substring(0, 7).replace("-", "")
    })
    df = df.withColumn("year_month", computeYearMonth(df("time")))
    // group by
    val columnsToAgg: Array[String] = Array("tg", "tn", "tx", "pp", "rr")
    val groupOn: Seq[String] = Seq("longitude", "latitude", "year_month")
    val grouped: DataFrame = df.groupBy(groupOn.head, groupOn.drop(1): _*)
      .agg(columnsToAgg.map(column => column -> "mean").toMap)
    // join
    df = df.join(grouped, groupOn)
    df = df.drop("year_month")

    // final evaluate
    if (options.contains('check)) {
      // want a single output file which means all data must fit in the memory of 1 machine;
      // for this experiment, this is expected
      df.coalesce(1)
        .withColumnRenamed("avg(tg)", "tg_mean")
        .withColumnRenamed("avg(tn)", "tn_mean")
        .withColumnRenamed("avg(tx)", "tx_mean")
        .withColumnRenamed("avg(rr)", "rr_mean")
        .withColumnRenamed("avg(pp)", "pp_mean")
        .write
        .option("header", "true")
        .csv(options('output) + "result")
    } else {
      println(df.collect())
    }

    // final rename of output csv's
    if (options.contains('check)) {
      val fs = FileSystem.get(spark.sparkContext.hadoopConfiguration)
      val files: List[String] = List("head", "agg", "result")
      for (f <- files) {
        val file = fs.globStatus(new Path(options('output) + f + "/part*"))(0).getPath.getName
        fs.rename(new Path(options('output) + f + "/" + file), new Path(options('output) + f + ".csv"))
        fs.delete(new Path(options('output) + f), true)
      }
    }

    spark.stop()
  }
}