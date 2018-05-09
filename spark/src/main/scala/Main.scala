import java.io.IOException

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.{FloatType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.sql.functions.{udf, abs}
import ucar.nc2.time.CalendarDate
import ucar.nc2.{NetcdfFile, Variable}

import collection.JavaConversions._


// TODO: fat jar http://blog.prabeeshk.com/blog/2014/04/08/creating-uber-jar-for-spark-project-using-sbt-assembly/
object Main {
  private val PATH: String = "/export/scratch1/radujica/datasets/ECAD/original/small_sample/"
  //private val PATH: String = System.getenv("HOME2") + "/datasets/ECAD/original/small_sample/"
  private val CALENDAR: String = "proleptic_gregorian"
  private val UNITS: String = "days since 1950-01-01"

  // read everything as floats to allow easy transpose
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

  // TODO: this is VERY ugly
  private def cartesianProductDimensions(lon: Array[Float], lat: Array[Float], tim: Array[Float]) = {
    val dims = Array("longitude", "latitude", "time")
    val newSize = lon.length * lat.length * tim.length

    // inner x2
    val newLon = new Array[Float](newSize)
    var times = lat.length * tim.length
    var index = 0
    for (aLon <- lon) {
      var j = 0
      while (j < times) {
        newLon(index) = aLon
        index += 1
        j += 1
      }
    }

    // outer + inner
    val newLatTemp = new Array[Float](lon.length * lat.length)
    times = lon.length
    index = 0
    var i = 0
    while (i < times) {
      for (aLat <- lat) {
        newLatTemp(index) = aLat
        index += 1
      }
      i += 1
    }
    val newLat = new Array[Float](newSize)
    times = tim.length
    index = 0
    for (aLat <- newLatTemp) {
      var j = 0
      while (j < times) {
        newLat(index) = aLat
        index += 1
        j += 1
      }
    }

    // outer x2
    val newTim = new Array[Float](newSize)
    times = lon.length * lat.length
    index = 0
    i = 0
    while (i < times) {
      for (aTim <- tim) {
        newTim(index) = aTim
        index += 1
      }
      i += 1
    }

    Array(newLon, newLat, newTim)
  }

  // TODO: generalize this
  // TODO: create rdd for each column and union? this avoids having all the data in driver memory at once
  // but shuffles data to workers n columns times; 1) do we have setup small cluster to use? pointless otherwise
  // 2. is transpose more or less costly than this?
  @throws[IOException]
  private def readData(path: String, ss: SparkSession, size: Int, addIndex: Boolean): DataFrame = {
    val file = NetcdfFile.open(path)
    var sizeF = size
    if (addIndex) sizeF += 1
    val data: Array[Array[Float]] = new Array[Array[Float]](sizeF)
    val names: Array[String] = new Array[String](sizeF)
    var i = 0
    for (variable : Variable <- file.getVariables) {
      data(i) = readVariable(variable)
      names(i) = variable.getShortName
      i += 1
    }

    // cartesian product for dimensions
    val dimensions = cartesianProductDimensions(data(names.indexOf("longitude")),
      data(names.indexOf("latitude")),
      data(names.indexOf("time")))
    data(names.indexOf("longitude")) = dimensions(0)
    data(names.indexOf("latitude")) = dimensions(1)
    data(names.indexOf("time")) = dimensions(2)

    // add index column (needed for subset by row numbers)
    if (addIndex) {
      data(i) = Array.tabulate(data(0).length)(_ + 1)
      names(i) = "index"
    }

    val rdd: RDD[Row] = ss.sparkContext.parallelize(data.transpose.toSeq.map(x => Row.fromSeq(x.toSeq)))

    val df: DataFrame = ss.createDataFrame(rdd, StructType(names.map(StructField(_, FloatType, nullable = false)).toSeq))

    // convert time from float to String
    val floatTimeToString = udf((time: Float) => {
      val udunits = String.valueOf(time.asInstanceOf[Int]) + " " + UNITS

      CalendarDate.parseUdunits(CALENDAR, udunits).toString.substring(0, 10)
    })
    df.withColumn("time", floatTimeToString(df("time")))
  }

  def main(args: Array[String]): Unit = {
    val spark: SparkSession = SparkSession.builder
      .appName("Spark Pipeline")
      .master("local[*]")  // local or local[4]
      //.config("spark.executor.heartbeatInterval", 20)
      .config("spark.executor.memory", "2g")
      .config("spark.driver.memory", "16g")
      .config("spark.driver.maxResultSize", "10g")
      .getOrCreate()

    // TODO: optimize partitioning ~ same partitioner in paralellize
    val df1: DataFrame = readData(PATH + "data1.nc", spark, 9, addIndex = true)
    val df2: DataFrame = readData(PATH + "data2.nc", spark, 7, addIndex = false)

    // PIPELINE
    // 1. join the 2 dataframes
    var df: DataFrame = df1.join(df2, Seq("longitude", "latitude", "time"), "inner")

    // 2. quick preview on the data
    println(df.show(10))

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
    println(df.drop("longitude", "latitude", "time").describe().show())
    // describe also computes count but presumably the aggregations computation is optimized;
    // doing the aggregations separately (as below) is more costly
//    val aggregations = Seq("min", "max", "mean", "std")
//    val columnsToAgg = df.drop("longitude", "latitude", "time").columns
//    for (aggregation: String <- aggregations) {
//      val aggr_map: Map[String, String] = columnsToAgg.map(column => column -> aggregation).toMap
//      println(df.agg(aggr_map).show())
//    }

    // 8. compute mean per month
    // UDF 2: compute custom year+month format
    val computeYearMonth = udf((time: String) => {
      time.substring(0, 7).replace("-", "")
    })
    df = df.withColumn("year_month", computeYearMonth(df("time")))
    // group by
    val columnsToAgg: Array[String] = Array("tg", "tn", "tx", "pp", "rr")
    val groupOn: Seq[String] = Seq("longitude", "latitude", "year_month")
    val grouped: DataFrame = df.groupBy(groupOn.head, groupOn.drop(1): _*).agg(columnsToAgg.map(column => column -> "mean").toMap)
    // join
    df = df.join(grouped, groupOn)
    df = df.drop("year_month")

    // final evaluate
    println(df.collect())

    spark.stop()
  }
}