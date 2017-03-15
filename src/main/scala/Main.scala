import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.feature.HashingTF
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.{SparkConf, SparkContext}

object Main extends App {

  val conf = new SparkConf()
    .setAppName("DatasetSpam")
    .setMaster("local")

  val sc = new SparkContext(conf)

  val hashingTf = new HashingTF(1024)

  val data = sc.textFile("dataset_spam.csv").map { line =>
    line.split(",").toList match {
      case "0" :: _ => LabeledPoint(0.0, hashingTf.transform(line.substring(2).split(" ")))
      case "1" :: _ => LabeledPoint(1.0, hashingTf.transform(line.substring(2).split(" ")))
    }
  }


  val Array(training, test) = data.randomSplit(Array(0.6, 0.4))

  val model = NaiveBayes.train(training, lambda = 1.0)

  val predictionAndLabel = test.map(p => (model.predict(p.features), p.label))

  val accuracy = 1.0 * predictionAndLabel.filter(x => x._1 == x._2).count() / test.count()

  println(s"accuracy $accuracy")

  //------

  val madeUpSpam = List(
    "watch lemons in hyderabad",
    "comic radiation dataset",
    "molecular cell 233",
    "bananas",
    "washing machine repairs",
    "buy washing",
    "washing repairs",
    "buy washing machine",
    "watch logan now"

  )
  val testSpam = madeUpSpam.map(name => (name, hashingTf.transform(name.mkString.split(" "))))
  testSpam.foreach {
    case (name, prediction) => println(s"$name ===> ${model.predict(prediction)}")
  }

}
