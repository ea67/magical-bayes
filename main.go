package main

import (
	"fmt"
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

	bayesBrain.Learn("Chinese", "Chinese", "Beijing", "Chinese")
	bayesBrain.Learn("Chinese", "Chinese", "Chinese", "Shanghai")
	bayesBrain.Learn("Chinese", "Chinese", "Macao")
	bayesBrain.Learn("Not Chinese", "Tokyo", "Japan", "Chinese")

	bayesBrain.Show()
	bayesBrain.ApplyTfIdf()
	bayesBrain.Show()
	features := []string{"Chinese", "Chinese", "Chinese", "Tokyo", "Japan"}


	fmt.Println(bayesClassifier.BayesProbabilityOf("Chinese",
		features...))
	fmt.Println(bayesClassifier.BayesProbabilityOf("Not Chinese",
		features...))

	fmt.Println(bayesClassifier.Classify(features...))


}
