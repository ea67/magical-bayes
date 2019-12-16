package brain

import (
	"fmt"
	"github.com/json-iterator/go"
	"io/ioutil"
	"math"
	"os"
	"path/filepath"
)

var VERSION = "v0.0.1"

type BayesBrain struct {
	FeaturesFrequency               map[string]float64
	CategoriesFrequency             map[string]float64
	FeaturesFrequencyInEachCategory map[string]map[string]float64
	CategoriesSummary               map[string]*CategorySummary
	LearnedCount                    int
	DidConvertTfIdf   				bool
	TfIdfTempValues					map[string]map[string]float64
}

type CategorySummary struct {
	Tfs          map[string][]float64
	LearnedCount int
}

func NewBayesBrain() *BayesBrain{
	 brain := new(BayesBrain)
	 brain.FeaturesFrequency = make(map[string]float64)
	 brain.CategoriesFrequency = make(map[string]float64)
	 brain.FeaturesFrequencyInEachCategory = make(map[string]map[string]float64)
	 brain.CategoriesSummary = make(map[string]*CategorySummary)
	 brain.LearnedCount = 0
	 return brain
}

func learn(featuresFrequency map[string]float64,  features []string) {
	for _, feature := range features {
		featuresFrequency[feature]++
	}
}

// TF-IDF https://en.wikipedia.org/wiki/Tf%E2%80%93idf
func (brain *BayesBrain) applyTfIdf() {

	if brain.DidConvertTfIdf {
		panic("Cannot call applyTfIdf more than once. Reset and relearn to reconvert.")
	}

	for category := range brain.CategoriesSummary {
		for feature := range brain.CategoriesSummary[category].Tfs {
			tfIdfSum := float64(0)
			for i, tf := range brain.CategoriesSummary[category].Tfs[feature] {
				brain.CategoriesSummary[category].Tfs[feature][i] = math.Log1p(tf) * math.Log1p(float64(brain.LearnedCount)/float64(brain.CategoriesSummary[feature].LearnedCount))
				tfIdfSum += brain.CategoriesSummary[category].Tfs[feature][i]
			}
			//brain.CategoriesSummary[category].t
			brain.TfIdfTempValues[category][feature] = tfIdfSum
		}

	}


	brain.DidConvertTfIdf = true
}

func (brain *BayesBrain) Learn(category string, features ...string) {
	learn(brain.FeaturesFrequency, features)
	brain.CategoriesFrequency[category]++
	if brain.FeaturesFrequencyInEachCategory[category] == nil {
		brain.FeaturesFrequencyInEachCategory[category] = make(map[string]float64)
	}
	learn(brain.FeaturesFrequencyInEachCategory[category], features)

	//tf-idf
	if brain.CategoriesSummary[category] == nil {
		brain.CategoriesSummary[category] =  new(CategorySummary)
	}
	brain.CategoriesSummary[category].LearnedCount++

	tfs := make(map[string]float64)
	for _, feature := range features {
		tfs[feature]++
		if brain.CategoriesSummary[category].Tfs[feature] == nil {
			brain.CategoriesSummary[category].Tfs = make(map[string][]float64)
		}
	}
	featureCount := float64(len(features))


	for feature, count := range tfs {

		tfs[feature] = count / featureCount
		// add the TF sample, after training we can get IDF values.
		brain.CategoriesSummary[category].Tfs[feature] = append(brain.CategoriesSummary[category].Tfs[feature], tfs[feature])
	}

	brain.LearnedCount++


}

func (brain *BayesBrain) Show() {
	fmt.Println("~~~~~~~~~~~ Bayes Brain " + VERSION + " ~~~~~~~~~~~")
	fmt.Println(brain.FeaturesFrequency)
	fmt.Println(brain.CategoriesFrequency)
	fmt.Println(brain.FeaturesFrequencyInEachCategory)
	fmt.Println(brain.LearnedCount)
	fmt.Println("tf-idf")
	categoriesSummary, err := jsoniter.ConfigCompatibleWithStandardLibrary.Marshal(brain.CategoriesSummary)
	if err != nil {
		panic(err)
	}
	fmt.Println(string(categoriesSummary))
	fmt.Println(brain.LearnedCount)
	fmt.Println("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
}

func (brain *BayesBrain) Save(filename string) error {
	if filename[len(filename):] != "/" {
		filename += "/"
	}
	parentPath := filename + VERSION
	err := save(brain.CategoriesFrequency, parentPath+"/cf.json")
	if err != nil {
		return err
	}
	err = save(brain.FeaturesFrequency, parentPath+"/ff.json")
	if err != nil {
		return err
	}
	err = save(brain.FeaturesFrequencyInEachCategory, parentPath+"/ffiec.json")
	if err != nil {
		return err
	}
	return nil
}

func save(obj interface{}, filename string) error {
	jsonBytes, err := jsoniter.ConfigCompatibleWithStandardLibrary.Marshal(obj)
	if err != nil {
		return err
	}

	path := filepath.Dir(filename)

	if _, err := os.Stat(path); os.IsNotExist(err) {
		err := os.MkdirAll(path, os.ModePerm)
		if err != nil {
			return err
		}
	}
	return ioutil.WriteFile(filename, jsonBytes, 0600)
}

func (brain *BayesBrain) Load(filename string) error {
	if filename[len(filename):] != "/" {
		filename += "/"
	}
	parentPath := filename + VERSION
	jsonBytes, err := load(parentPath + "/cf.json")
	if err != nil {
		return err
	}
	err = jsoniter.ConfigCompatibleWithStandardLibrary.Unmarshal(jsonBytes, &brain.CategoriesFrequency)
	if err != nil {
		return err
	}

	jsonBytes, err = load(parentPath + "/ff.json")
	if err != nil {
		return err
	}
	err = jsoniter.ConfigCompatibleWithStandardLibrary.Unmarshal(jsonBytes, &brain.FeaturesFrequency)
	if err != nil {
		return err
	}


	jsonBytes, err = load(parentPath + "/ffiec.json")
	if err != nil {
		return err
	}
	err = jsoniter.ConfigCompatibleWithStandardLibrary.Unmarshal(jsonBytes, &brain.FeaturesFrequencyInEachCategory)
	if err != nil {
		return err
	}

	return nil
}

func load(filename string) ([]byte, error) {
	content, err := ioutil.ReadFile(filename)
	if err != nil {
		return nil, err
	}
	return content, nil

}