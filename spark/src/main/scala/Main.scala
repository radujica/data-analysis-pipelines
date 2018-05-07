import org.apache.spark.sql.SparkSession

object Main {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder
      .appName("Spark Pipeline")
      .master("local")
      .getOrCreate()

    println("hello")

    spark.stop()
  }
}