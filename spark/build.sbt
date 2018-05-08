scalaVersion := "2.11.8"

name := "spark"
version := "1.0"

resolvers += "Unidata maven repository" at "http://artifacts.unidata.ucar.edu/content/repositories/unidata-releases"
libraryDependencies += "org.apache.spark" %% "spark-sql" % "2.3.0"
libraryDependencies += "edu.ucar" % "netcdf" % "4.3.22"
libraryDependencies += "org.typelevel" %% "cats-core" % "0.9.0"
