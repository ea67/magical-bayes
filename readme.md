
#### 快速入门

```go
package main

import (
	"fmt"
	"magicalbayes/bayes/brain"
	"magicalbayes/bayes/classifier"
)

func main()  {
	//实例化 贝叶斯大脑
    bayesBrain := brain.NewBayesBrain()
    //构造分类器
    bayesClassifier := classifier.BayesClassifier{Brain: bayesBrain}
    //训练样本数据
    bayesBrain.Learn("Chinese", "Chinese", "Beijing", "Chinese")
    bayesBrain.Learn("Chinese", "Chinese", "Chinese", "Shanghai")
    bayesBrain.Learn("Chinese", "Chinese", "Macao")
    bayesBrain.Learn("Not Chinese", "Tokyo", "Japan", "Chinese")
    bayesBrain.Show()

    //测试样本数据
    features := []string{"Chinese", "Chinese", "Chinese", "Tokyo", "Japan"}

    //计算属于该类别的概率
    fmt.Println(bayesClassifier.BayesProbabilityOf("Chinese",
        features...))
    fmt.Println(bayesClassifier.BayesProbabilityOf("Not Chinese",
        features...))
    //对测试样本进行分类
    fmt.Println(bayesClassifier.Classify(features...))
}


```

#### 待更新...