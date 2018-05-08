import java.io.IOException

import org.apache.spark.rdd.RDD
import org.apache.spark.sql.types.{FloatType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.sql.functions.udf
import ucar.nc2.time.CalendarDate
import ucar.nc2.{NetcdfFile, Variable}

import collection.JavaConversions._

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
  @throws[IOException]
  private def readData(path: String, ss: SparkSession, size: Int): DataFrame = {
    val file = NetcdfFile.open(path)
    //var data: mutable.LinkedHashMap[String, Array[Float]] = new mutable.LinkedHashMap[String, Array[Float]]()
    val data: Array[Array[Float]] = new Array[Array[Float]](size)
    val names: Array[String] = new Array[String](size)
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

    val rdd: RDD[Row] = ss.sparkContext.parallelize(data.transpose.toSeq.map(x => Row.fromSeq(x.toSeq)))

    val df: DataFrame = ss.createDataFrame(rdd, StructType(names.map(StructField(_, FloatType, nullable = false)).toSeq))

    // convert time from (float) to String
    val floatTimeToString = udf((time: Float) => {
      val udunits = String.valueOf(time.asInstanceOf[Int]) + " " + UNITS

      CalendarDate.parseUdunits(CALENDAR, udunits).toString.substring(0, 10)
    })
    df.withColumn("time", floatTimeToString(df("time")))
  }

  def main(args: Array[String]): Unit = {
    val spark: SparkSession = SparkSession.builder
      .appName("Spark Pipeline")
      .master("local")
      .config("spark.executor.heartbeatInterval", 20)
      .getOrCreate()

    val df1: DataFrame = readData(PATH + "data1.nc", spark, 9)
    val df2: DataFrame = readData(PATH + "data2.nc", spark, 7)

    val df = df1.join(df2, Seq("longitude", "latitude", "time"), "inner")

    spark.stop()
  }
}