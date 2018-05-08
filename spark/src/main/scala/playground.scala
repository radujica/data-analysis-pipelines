import scala.collection.mutable.ListBuffer


object PlayGround {

  def crossJoin[T](list: Traversable[Traversable[Float]]): Traversable[Traversable[Float]] =
    list match {
      case xs :: Nil => xs map (Traversable(_))
      case x :: xs => for {
        i <- x
        j <- crossJoin(xs)
      } yield Traversable(i) ++ j
    }

  def main(args: Array[String]): Unit = {
    //val l = new ListBuffer[Array[_]]()
//        val arr1 = new Array[Float](2)
//        arr1(0) = 12f
//        arr1(1) = 13f
//    val arr2 = new Array[Float](2)
//    arr2(0) = 14f
//    arr2(1) = 15f
//    println(arr1)
//    println(Traversable(arr1).head)
//    println((arr1.map(Traversable(_)) ++ arr2.map(Traversable(_))).length)
//    val res: Array[Traversable[Float]] = arr1.map(Traversable(_))
//    println(res.head)
//    println(crossJoin(List(arr1, arr2)))


//    val l = new ListBuffer[Array[_]]()
//    val arr1 = new Array[Int](2)
//    arr1(0) = 12
//    arr1(1) = 13
//    l += arr1
//    val arr2 = new Array[Float](2)
//    arr2(0) = 10.0f
//    arr2(1) = 11.0f
//    l += arr2
//
//    println(l.transpose)

//    val l = new ListBuffer[Array[Int]]()
//    val arr1 = new Array[Int](2)
//    arr1(0) = 12
//    arr1(1) = 13
//    l += arr1
//    val arr2 = new Array[Int](2)
//    arr2(0) = 10
//    arr2(1) = 11
//    l += arr2
//
//    println(l.transpose)
  }

}