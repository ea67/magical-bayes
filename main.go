package main

import (
	"fmt"
	"github.com/jbrukh/bayesian"
	"magicalbayes/bayes/brain"
	"magicalbayes/bayes/classifier"
)

func main()  {
 	test0()

}

func test0()  {

	bayesBrain := brain.NewBayesBrain()
	bayesClassifier := classifier.BayesClassifier{Brain: bayesBrain}
	bayesBrain.Show()

	err := bayesBrain.Load("/Users/liumingxing/go/src/magicalbayes/db")
	if err!= nil{
		panic(err)
	}
	//
	//bayesBrain.Learn("Chinese", "Chinese", "Beijing", "Chinese")
	//fmt.Println(bayesBrain.CategoriesSummary["Chinese"],",,,,,,,,")
	//bayesBrain.Learn("Chinese", "Chinese", "Chinese", "Shanghai")
	//fmt.Println(bayesBrain.CategoriesSummary["Chinese"],",,,,,,,,")
	//bayesBrain.Learn("Chinese", "Chinese", "Macao")
	//bayesBrain.Learn("Not Chinese", "Tokyo", "Japan", "Chinese")

	bayesBrain.Show()
	bayesBrain.ApplyTfIdf()
	bayesBrain.Show()
	features := []string{"Chinese", "Chinese", "Chinese", "Tokyo", "Japan"}


	fmt.Println(bayesClassifier.BayesProbabilityOf("Chinese",
		features...))
	fmt.Println(bayesClassifier.BayesProbabilityOf("Not Chinese",
		features...))

	fmt.Println(bayesClassifier.Classify(features...))

	//err2 := bayesBrain.Save("/Users/liumingxing/go/src/magicalbayes/db")
	//if err2!= nil {
	//	panic(err2)
	//
	//}

	return
	c := bayesian.NewClassifierTfIdf(Good, Bad)

	c.Learn([]string{"tall", "handsome", "rich"}, Good)
	c.Learn([]string{"tall", "blonde"}, Good)
	c.Learn([]string{"tall"}, Good)
	c.Learn([]string{"fat"}, Bad)
	c.Learn([]string{"short", "poor"}, Bad)


	c.ConvertTermsFreqToTfIdf()

	score, likely, strict := c.LogScores([]string{"the", "tall", "man"})
	fmt.Printf("%#v", score)
	fmt.Printf("%#v", likely)
	fmt.Printf("%#v", strict)
}
const (
	Good bayesian.Class = "good"
	Bad  bayesian.Class = "bad"
)


//category  Chinese  feature  Chinese   1.209192634657479
//category  Chinese  feature  Beijing   0.24375240446015042
//category  Chinese  feature  Shanghai   0.24375240446015042
//category  Chinese  feature  Macao   0.34354971856171385
//category  Not Chinese  feature  Tokyo   0.4630064341315098
//category  Not Chinese  feature  Japan   0.4630064341315098
//category  Not Chinese  feature  Chinese   0.4630064341315098

//category  Chinese  feature  Chinese   0.8596737056461582
//category  Chinese  feature  Beijing   0.18482357564923738
//category  Chinese  feature  Shanghai   0.18482357564923738
//category  Chinese  feature  Macao   0.2502198990845568
//category  Not Chinese  feature  Tokyo   0.6123806964123236
//category  Not Chinese  feature  Japan   0.6123806964123236
//category  Not Chinese  feature  Chinese   0.6123806964123236


//category  Not Chinese  feature  Tokyo   0.4630064341315098
//category  Not Chinese  feature  Japan   0.4630064341315098
//category  Not Chinese  feature  Chinese   0.4630064341315098
//category  Chinese  feature  Macao   0.34354971856171385
//category  Chinese  feature  Chinese   1.209192634657479
//category  Chinese  feature  Beijing   0.24375240446015042
//category  Chinese  feature  Shanghai   0.24375240446015042
//
//category  Not Chinese  feature  Chinese   0.6123806964123236
//category  Not Chinese  feature  Japan   0.6123806964123236
//category  Not Chinese  feature  Tokyo   0.6123806964123236
//category  Chinese  feature  Beijing   0.18482357564923738
//category  Chinese  feature  Chinese   0.8596737056461582
//category  Chinese  feature  Macao   0.2502198990845568
//category  Chinese  feature  Shanghai   0.18482357564923738
