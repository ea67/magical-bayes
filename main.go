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
	bayesBrain.Learn("Chinese", "Chinese", "Beijing", "Chinese")
	bayesBrain.Learn("Chinese", "Chinese", "Chinese", "Shanghai")
	bayesBrain.Learn("Chinese", "Chinese", "Macao")
	bayesBrain.Learn("Not Chinese", "Tokyo", "Japan", "Chinese")
	bayesBrain.Show()

	features := []string{"Chinese", "Chinese", "Chinese", "Tokyo", "Japan"}

	fmt.Println(bayesClassifier.BayesProbabilityOf("Chinese",
		features...))
	fmt.Println(bayesClassifier.BayesProbabilityOf("Not Chinese",
		features...))

	fmt.Println(bayesClassifier.Classify(features...))


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